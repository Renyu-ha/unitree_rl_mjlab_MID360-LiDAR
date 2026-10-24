# Solver contact regressions

`go2w_linesearch_limit.npz` and `go2w_solver_limit.npz` contain single-world,
float32 inputs captured before an actual `forward` or `step` call in the complete
`Unitree-Go2W-Perceptive-Rough-v2` task. They contain no recomputed failure output.

The model is reconstructed with seed 42 and ten difficulty rows. Curriculum
generation creates seven terrain columns (one per terrain type), regardless of
the preset's requested six columns. Frozen replay restores the original 3 m
stair platforms and 1 mm wheel margins; current training uses 2 m platforms and
zero wheel margins. The files include qpos, qvel, control, warmstart, applied forces,
time and the nonempty per-world domain-randomized model fields. Empty tendon
fields are intentionally omitted. The model has 23 qpos, 22 velocities, 16
actuators and 446 geoms. No MJB or installed dependency file is required.

Captured on CPU with MuJoCo/MuJoCo-Warp 3.16.0 and Warp 1.18.0:

- Linesearch case: seed 42, 8 worlds, unit Gaussian actions; world 5 before a
  reset `forward` (physics-call index 81). Budget 10/20 sets LS_ITERATIONS.
- Solver case: seed 42, 32 worlds, Gaussian actions with standard deviation 2;
  world 30 before a `step` (physics-call index 917). Budget 10/20 sets ITERATIONS.

The test verifies each original failure, checks that the task's 50/100 budget
clears both diagnostic flags, and compares accelerations against a 100/200
reference with the same tolerance, inputs, contacts and randomized model fields.
This is a fixed-state regression, not proof that every possible GPU contact or
long training run converges within a finite budget.

These inputs predate the terrain-clearance reset guard. Replay intentionally
restores the original frozen state to retain the solver-budget regression.
