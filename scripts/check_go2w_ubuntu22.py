"""Run the real C++ controller and original unitree_mujoco over loopback DDS."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]


def stop(process):
  if process.poll() is None:
    process.send_signal(signal.SIGINT)
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait(timeout=5)


def run_case(name, args, base_domain, controller, simulator, policy, output, relaxed=False):
  directory=output/name
  directory.mkdir(parents=True,exist_ok=True)
  config=directory/"controller.yaml"
  hardware=(ROOT/"deploy/robots/go2w/config/config.yaml").read_text()
  if relaxed:
    for before,after in [("lowstate_timeout_ms: 100","lowstate_timeout_ms: 2000"),
                         ("command_timeout_ms: 50","command_timeout_ms: 2000"),
                         ("policy_timeout_ms: 60","policy_timeout_ms: 2000"),
                         ("pointcloud_timeout_ms: 300","pointcloud_timeout_ms: 2000"),
                         ("odometry_timeout_ms: 200","odometry_timeout_ms: 2000")]:
      hardware=hardware.replace(before,after)
  config.write_text(hardware)
  duration=12 if relaxed else 9
  sim_command=[str(simulator),"--config",str(ROOT/"simulate/config_go2w.yaml"),"--headless","--auto",
    f"--domain_id={base_domain}",f"--duration={duration}","--report",str(directory/"simulation.json")]+args
  ctrl_command=[str(controller),"--sim","--network=lo",f"--domain={base_domain}","--policy",str(policy),"--config",str(config)]
  if name!="readonly": ctrl_command.append("--actuate")
  with (directory/"simulator.log").open("w") as sim_log,(directory/"controller.log").open("w") as ctrl_log:
    sim=subprocess.Popen(sim_command,stdout=sim_log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+(90 if relaxed else 20)
    while "Go2W headless" not in (directory/"simulator.log").read_text():
      if sim.poll() is not None or time.monotonic()>deadline:
        stop(sim)
        raise RuntimeError(f"{name}: simulator did not become ready")
      time.sleep(0.05)
    ctrl=subprocess.Popen(ctrl_command,stdout=ctrl_log,stderr=subprocess.STDOUT)
    try:
      sim.wait(timeout=180 if relaxed else 40)
      if sim.returncode:
        detail=(directory/"simulator.log").read_text()[-2000:]
        raise RuntimeError(f"{name}: simulator exited {sim.returncode}: {detail}")
      if ctrl.poll() is not None: raise RuntimeError(f"{name}: controller exited {ctrl.returncode}")
    finally:
      stop(ctrl); stop(sim)
  text=(directory/"controller.log").read_text()
  report=json.loads((directory/"simulation.json").read_text())
  if name=="stand_estop":
    if "Change state from FixStand to Passive" not in text or "Go2W policy steps=" in text:
      raise RuntimeError("Stop during standing did not remain Passive")
  elif "Go2W policy steps=" not in text: raise RuntimeError(f"{name}: policy never executed")
  if name in {"cloud_timeout","odometry_timeout","lowstate_timeout","crc_timeout"}:
    if "Change state from Velocity to Passive" not in text:
      raise RuntimeError(f"{name}: fault did not return to Passive")
  elif "Change state from Velocity to Passive" in text:
    raise RuntimeError(f"{name}: unexpected policy fault")
  if report["max_wheel_kp"]!=0: raise RuntimeError(f"{name}: wheel position gain is nonzero")
  if name=="readonly" and report["commands"]!=0: raise RuntimeError("Read-only controller published LowCmd")
  if name!="readonly" and report["commands"]<=0: raise RuntimeError(f"{name}: no motor commands reached simulator")
  if name=="normal" and report["min_base_height"]<0.2: raise RuntimeError("Zero-action standing was unstable")
  if name=="wheel_mapping" and any(abs(a-b)>1e-4 for a,b in zip(report["wheel_targets_sdk"],[2,1,4,3])):
    raise RuntimeError(f"Incorrect wheel SDK order: {report['wheel_targets_sdk']}")
  report["passed"]=True
  print(f"PASS {name}",flush=True)
  return report


def check_startup_guards(args, domain):
  directory=args.output/"startup_guards"
  directory.mkdir(parents=True,exist_ok=True)
  with (directory/"simulator.log").open("w") as log:
    sim=subprocess.Popen([str(args.simulator.resolve()),"--config",str(ROOT/"simulate/config_go2w.yaml"),
      "--headless","--freeze-physics",f"--domain_id={domain}","--duration=120"],stdout=log,stderr=subprocess.STDOUT)
    owner=None
    try:
      deadline=time.monotonic()+90
      while "Go2W headless" not in (directory/"simulator.log").read_text():
        if sim.poll() is not None or time.monotonic()>deadline: raise RuntimeError("Startup guard simulator did not start")
        time.sleep(0.05)
      common=[str(args.controller.resolve()),"--network=lo",f"--domain={domain}","--config",str(ROOT/"deploy/robots/go2w/config/config.yaml")]
      smoke=ROOT/"tests/fixtures/go2w_deploy"
      result=subprocess.run(common+["--actuate","--policy",str(smoke)],capture_output=True,text=True,timeout=30)
      (directory/"smoke_rejection.log").write_text(result.stdout+result.stderr)
      if result.returncode!=1 or "smoke policy is for --sim only" not in result.stdout+result.stderr:
        raise RuntimeError("Hardware output accepted a simulation smoke policy")
      damaged=directory/"damaged_policy"
      shutil.copytree(smoke,damaged,dirs_exist_ok=True)
      yaml=damaged/"params/deploy.yaml"
      yaml.write_text(yaml.read_text().replace("observation_dim: 720","observation_dim: 719",1))
      result=subprocess.run(common+["--sim","--policy",str(damaged.resolve())],capture_output=True,text=True,timeout=30)
      (directory/"contract_rejection.log").write_text(result.stdout+result.stderr)
      if result.returncode!=1 or "differs from exported contract" not in result.stdout+result.stderr:
        raise RuntimeError("Controller accepted a modified deployment contract")
      with (directory/"owner.log").open("w") as owner_log:
        owner=subprocess.Popen(common+["--sim","--actuate","--policy",str(smoke)],stdout=owner_log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while "FSM: Start Passive" not in (directory/"owner.log").read_text():
          if owner.poll() is not None or time.monotonic()>deadline: raise RuntimeError("Owner controller did not start")
          time.sleep(0.05)
        time.sleep(0.1)
        result=subprocess.run(common+["--sim","--actuate","--policy",str(smoke)],capture_output=True,text=True,timeout=30)
        (directory/"duplicate_rejection.log").write_text(result.stdout+result.stderr)
        if result.returncode!=1 or "already has a writer" not in result.stdout+result.stderr:
          raise RuntimeError("Second motor-command writer was not rejected")
    finally:
      if owner: stop(owner)
      stop(sim)
  print("PASS startup_guards (smoke / contract / duplicate writer)",flush=True)
  return {"passed":True,"smoke_policy_rejected":True,"modified_contract_rejected":True,"duplicate_writer_rejected":True}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--controller",type=Path,default=ROOT/"deploy/robots/go2w/build/go2w_ctrl")
  parser.add_argument("--simulator",type=Path,default=ROOT/"simulate/build/unitree_mujoco")
  parser.add_argument("--policy",type=Path,default=ROOT/"tests/fixtures/go2w_deploy")
  parser.add_argument("--output",type=Path,default=ROOT/"outputs/go2w_selfcheck")
  parser.add_argument("--relaxed-deadlines",action="store_true",help="For slow emulated test environments; not a hardware configuration")
  parser.add_argument("--case",action="append",choices=["startup_guards","normal","readonly","cloud_timeout","odometry_timeout","lowstate_timeout","crc_timeout","wheel_mapping","stand_estop"],help="Run selected checks; default: all")
  args=parser.parse_args()
  args.output.mkdir(parents=True,exist_ok=True)
  for path in [args.controller,args.simulator,args.policy/"exported/policy.onnx"]:
    if not path.exists(): parser.error(f"Missing {path}; build first")
  domain=30+os.getpid()%80
  results={}
  if not args.case or "startup_guards" in args.case:
    results["startup_guards"]=check_startup_guards(args,domain+10)
  cases=[("normal",[]),("readonly",["--freeze-physics"]),
    ("cloud_timeout",["--freeze-physics","--drop-cloud-after=6"]),("odometry_timeout",["--freeze-physics","--drop-odom-after=6"]),
    ("lowstate_timeout",["--freeze-physics","--drop-state-after=6"]),("crc_timeout",["--freeze-physics","--corrupt-state-after=6"]),
    ("wheel_mapping",["--freeze-physics"]),("stand_estop",["--freeze-physics","--estop-after=2"])]
  for index,(name,flags) in enumerate(cases):
    if args.case and name not in args.case: continue
    policy=ROOT/"tests/fixtures/go2w_wheel_probe" if name=="wheel_mapping" else args.policy
    results[name]=run_case(name,flags,domain+index,args.controller.resolve(),args.simulator.resolve(),policy.resolve(),args.output.resolve(),args.relaxed_deadlines)
  (args.output/"report.json").write_text(json.dumps(results,indent=2))
  print(f"All {len(results)} DDS checks passed. Reports: {args.output.resolve()}")


if __name__=="__main__":
  main()
