"""Exercise the upgraded Go2W sensors, physics, reset, PPO and checkpoint export.

Run after activating the Conda environment, e.g.:
  python scripts/check_go2w_upgrade.py --learn-iterations 2
  python scripts/check_go2w_upgrade.py --device cuda:0
"""
import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # noqa: F401


def check_finite(value, name):
  if isinstance(value, dict):
    for key,tensor in value.items():
      check_finite(tensor, f"{name}/{key}")
  elif isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
    raise RuntimeError(f"Non-finite values in {name}")


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--device',default='cpu')
  parser.add_argument('--num-envs',type=int,default=2)
  parser.add_argument('--steps',type=int,default=32)
  parser.add_argument('--learn-iterations',type=int,default=0)
  parser.add_argument('--output-dir',type=Path,default=Path('logs/upgrade_check'))
  args=parser.parse_args()
  if args.num_envs<2 or args.steps<1 or args.learn_iterations<0:
    parser.error('num-envs >= 2, steps >= 1 and learn-iterations >= 0 are required')
  task='Unitree-Go2W-Perceptive-Rough-v2'
  cfg=load_env_cfg(task)
  cfg.scene.num_envs=args.num_envs
  cfg.scene.terrain.terrain_generator.num_rows=3
  cfg.scene.terrain.terrain_generator.num_cols=2
  cfg.scene.terrain.max_init_terrain_level=0
  cfg.observations['actor'].nan_policy='error'
  cfg.observations['critic'].nan_policy='error'
  cfg.sim.nan_guard.enabled=True
  cfg.sim.nan_guard.output_dir=str(args.output_dir/'nan_dump')
  print({name:version(name) for name in ('mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','torch')},flush=True)
  env=ManagerBasedRlEnv(cfg=cfg,device=args.device)
  try:
    obs,_=env.reset()
    check_finite(obs,'reset')
    action=torch.zeros(env.action_space.shape,device=env.device)
    for step in range(args.steps):
      obs,reward,terminated,truncated,_=env.step(action)
      check_finite(obs,f'step_{step}')
      check_finite(reward,'reward')
      check_finite(env.sim.data.qacc,'qacc')
    qpos=env.sim.data.qpos[1:].clone()
    twist=env.command_manager.get_term('twist')
    # Heading commands can be refreshed from current kinematics on reset.
    # Their sampled targets, timers and counters must remain untouched.
    timers=twist.time_left[1:].clone()
    targets=twist.heading_target[1:].clone()
    counters=twist.command_counter[1:].clone()
    env.reset(env_ids=torch.tensor([0],device=env.device))
    torch.testing.assert_close(env.sim.data.qpos[1:],qpos,rtol=0,atol=0)
    torch.testing.assert_close(twist.time_left[1:],timers,rtol=0,atol=0)
    torch.testing.assert_close(twist.heading_target[1:],targets,rtol=0,atol=0)
    torch.testing.assert_close(twist.command_counter[1:],counters,rtol=0,atol=0)
    if args.learn_iterations:
      wrapped=RslRlVecEnvWrapper(env)
      rl=load_rl_cfg(task)
      rl.logger='tensorboard'
      rl.upload_model=False
      rl.num_steps_per_env=4
      rl.algorithm.num_mini_batches=1
      rl.algorithm.num_learning_epochs=1
      runner=load_runner_cls(task)(wrapped,asdict(rl),str(args.output_dir),args.device)
      runner.learn(args.learn_iterations,init_at_random_ep_len=False)
      checkpoint=args.output_dir/'model_check.pt'
      runner.save(str(checkpoint))
      runner.load(str(checkpoint),map_location=args.device)
    print('PASS: finite rollout, MID360/contact observations and isolated reset',flush=True)
  finally:
    env.close()


if __name__=='__main__':
  main()
