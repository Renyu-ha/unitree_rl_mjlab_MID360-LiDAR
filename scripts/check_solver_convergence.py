"""Check actual solver-limit bits on the training device, including reset forwards.

Example:
  python scripts/check_solver_convergence.py --device cuda:0 --num-envs 4096

This uses Gaussian actions, not a trained policy. Device reads synchronize every
physics call for diagnosis, so elapsed time is not a training throughput benchmark.
Warnings remain enabled. Any observed limit or capacity overflow fails the check.
"""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import time

import mujoco_warp as mjw
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import src.tasks  # noqa: F401


class SolverMonitor:
  """Count per-world diagnostics before the next reset can clear them."""

  def __init__(self, sim):
    self.sim = sim
    self.counts = {"world_checks": 0, "calls": 0, "linesearch_limit": 0,
                   "solver_limit": 0, "other_overflow": 0}
    self.max_iterations = 0
    self.examples = []
    self._original = {name: getattr(sim, name) for name in ("step", "forward")}

  def install(self):
    for name, operation in self._original.items():
      setattr(self.sim, name, self._wrap(name, operation))

  def restore(self):
    for name, operation in self._original.items():
      setattr(self.sim, name, operation)

  def _wrap(self, name, operation):
    def monitored(*args, **kwargs):
      # These are diagnostic bits, not physics inputs. Clear only after the
      # previous call has been counted, to avoid counting a sticky bit twice.
      self.sim.wp_data.overflow.zero_()
      result = operation(*args, **kwargs)
      flags = self.sim.wp_data.overflow.numpy()
      niter = self.sim.wp_data.solver_niter.numpy()
      ls_bit = int(mjw.OverflowType.LS_ITERATIONS)
      solver_bit = int(mjw.OverflowType.ITERATIONS)
      self.counts["calls"] += 1
      self.counts["world_checks"] += flags.size
      self.counts["linesearch_limit"] += int(np.count_nonzero(flags & ls_bit))
      self.counts["solver_limit"] += int(np.count_nonzero(flags & solver_bit))
      self.counts["other_overflow"] += int(np.count_nonzero(flags & ~(ls_bit | solver_bit)))
      self.max_iterations = max(self.max_iterations, int(niter.max()))
      if np.any(flags) and len(self.examples) < 5:
        ids = np.flatnonzero(flags)[:8]
        self.examples.append({"call": self.counts["calls"], "operation": name,
                              "env_ids": ids.tolist(), "flags": flags[ids].tolist(),
                              "solver_niter": niter[ids].tolist()})
      if not np.isfinite(self.sim.wp_data.qacc.numpy()).all():
        raise RuntimeError("Non-finite acceleration during solver check")
      return result
    return monitored

  @property
  def passed(self):
    return not any(self.counts[key] for key in
                   ("linesearch_limit", "solver_limit", "other_overflow"))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Unitree-Go2W-Perceptive-Rough-v2")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=256)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--action-std", type=float, default=1.0)
  parser.add_argument("--iterations", type=int)
  parser.add_argument("--ls-iterations", type=int)
  parser.add_argument("--output", type=Path, default=Path("logs/solver_check.json"))
  args = parser.parse_args()
  if args.num_envs < 1 or args.steps < 1 or args.action_std <= 0:
    parser.error("num-envs, steps and action-std must be positive")
  for budget in (args.iterations, args.ls_iterations):
    if budget is not None and budget < 1:
      parser.error("solver iteration budgets must be positive")
  cfg = load_env_cfg(args.task)
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  if args.iterations is not None:
    cfg.sim.mujoco.iterations = args.iterations
  if args.ls_iterations is not None:
    cfg.sim.mujoco.ls_iterations = args.ls_iterations
  env = ManagerBasedRlEnv(cfg, device=args.device)
  monitor = SolverMonitor(env.sim)
  report = {"task": args.task, "device": args.device, "seed": args.seed,
            "num_envs": args.num_envs, "steps": args.steps, "action_std": args.action_std,
            "scope": "reset and rollout after environment construction; inspect constructor graph-capture messages separately",
            "versions": {name: version(name) for name in
                         ("mjlab", "mujoco", "mujoco-warp", "warp-lang")},
            "mujoco": asdict(cfg.sim.mujoco), "passed": False}
  start = time.perf_counter()
  try:
    bits = int(mjw.OverflowType.LS_ITERATIONS | mjw.OverflowType.ITERATIONS)
    if env.sim.wp_model.opt.warn_overflow & bits != bits:
      raise RuntimeError("Solver-limit warnings must be enabled for this check")
    monitor.install()
    env.reset()
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for step in range(args.steps):
      action = args.action_std * torch.randn(
        env.action_space.shape, device=args.device, generator=generator)
      obs, reward, *_ = env.step(action)
      if not torch.isfinite(reward).all() or not all(torch.isfinite(x).all() for x in obs.values()):
        raise RuntimeError("Non-finite reward or observation")
      if (step + 1) % 64 == 0:
        print(f"[SolverCheck] {step + 1}/{args.steps}: {monitor.counts}", flush=True)
    report["passed"] = monitor.passed
  finally:
    monitor.restore()
    report.update(counts=monitor.counts, solver_niter_max=monitor.max_iterations,
                  examples=monitor.examples, diagnostic_seconds=time.perf_counter() - start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()
  print(json.dumps(report, indent=2), flush=True)
  if not report["passed"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
