"""Replay a mjlab NaN dump with visual/collision geometry group controls.

Example: python scripts/viz_nan.py logs/go2w_nan_dump/nan_dump_latest.npz
"""

import tyro

import mjlab
from mjlab.scripts.nan_viz import NanDumpViewer


def visualize(dump_path: tyro.conf.Positional[str]) -> None:
  viewer = NanDumpViewer(dump_path)
  try:
    viewer.setup()
    viewer.scene.create_groups_gui(viewer.server.gui.add_tab_group())
    viewer.server.gui.add_markdown(
      "Go2W collision inspection: in **Groups → Geoms**, disable **G2** "
      "(visual mesh) and enable **G3** (collision shapes). Keep **G0/G1** "
      "enabled for terrain. Use **Playback → Step** to inspect saved states.\n\n"
      "Contacts are recomputed by CPU MuJoCo from the saved state; they "
      "are not a recording of the original GPU contact forces."
    )
    viewer.run()
  finally:
    viewer.server.stop()


if __name__ == "__main__":
  tyro.cli(visualize, config=mjlab.TYRO_FLAGS)
