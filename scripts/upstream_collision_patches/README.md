# Pinned upstream EPA candidates

These are diagnostic backports of upstream changes, not locally invented
collision formulas or a claim of a released/general fix. Both PRs were open and
unmerged when checked on 2026-10-24.

- [PR 1771: Compute EPA witness on the face containing it](https://github.com/google-deepmind/mujoco_warp/pull/1771)
  - Base `b64d40c7764aa00d21bcb774b3ce423e002bfa1f`
  - Head `58dcd7c5158150a9f87e2359a488f4d6d8d7f0a9`
- [PR 1796: Never select EPA sliver faces from near-repeated support points](https://github.com/google-deepmind/mujoco_warp/pull/1796)
  - Base `c384ac51df67eb92ab7657cc472ff7494dc036bf`
  - Head `4833e00f83663cbc0964955bffdd0acb7f80bf2d`

Production patches are the exact differences between the pinned base/head
`mujoco_warp/_src/collision_gjk.py` files. The corresponding test hunks come from
the same upstream PRs. Copyright remains with the upstream authors under
Apache-2.0; the copied engine package retains its upstream headers and license.

`manifest.json` records commits and SHA-256 checksums, normalized to LF. The
baseline hash is the official v3.16.0 file, also identical to PR 1771's base
file. Applying 1771 reconstructs its head file exactly. PR 1796's two production
hunks apply unchanged to v3.16.0 and alongside 1771; other changes between its
base and v3.16.0 are not imported. No fuzzy or partial application is allowed.

`check_upstream_collision_fix.py` copies the installed package to a new output
directory and launches fresh child processes with those private copies on
`PYTHONPATH`. Package versions, the loaded source path and collision source
checksum are checked. All variants receive the same three upstream regression
cases, so baseline failures are retained as controls. FMA remains enabled and
EPA horizon stays 24. Installed packages and normal training are not modified.

A frozen-scene candidate requires finite physics, zero diagnostic flags, sane
nonempty contacts and agreement with native MuJoCo for contact pairs/counts,
position (1 mm), distance (50 micrometers) and normal angle (5 degrees). This
rejects the observed FMA-off result that removed all contacts. If the baseline
does not reproduce, the tool reports the comparison as inconclusive. A valid
candidate must also pass all three upstream regression cases, the project's
regression suite and a short 32-world Go2W rollout with two PPO iterations
before the overall check passes.
These are bounded checks, not proof of long-run stability or every task/terrain.

Do not install these candidates into the training environment automatically.
After GPU verification, fix the chosen dependency to a reviewed, reproducible
source revision and rerun task checks before updating production metadata.
