"""Replay a mjlab NaN dump with visual/collision geometry group controls.

Example: python scripts/viz_nan.py logs/go2w_nan_dump/nan_dump_latest.npz
"""

import tyro

import mjlab
from src.utils.border_inspection import add_border_inspection
from src.utils.nan_viewer import PhysicsNanDumpViewer


def visualize(dump_path: tyro.conf.Positional[str]) -> None:
  viewer = PhysicsNanDumpViewer(dump_path)
  try:
    viewer.setup()
    viewer.scene.create_groups_gui(viewer.server.gui.add_tab_group())
    add_border_inspection(viewer)
    viewer.server.gui.add_markdown(
      "Go2W collision inspection: in **Groups → Geoms**, disable **G2** "
      "(visual mesh) and enable **G3** (collision shapes). Keep **G0/G1** "
      "enabled for terrain. Use **Playback → Step** to inspect saved states.\n\n"
      "Scene contacts use CPU position reconstruction; solver force arrows "
      "are unavailable. New dumps "
      "include a terminal failure frame and raw failure_* arrays. When the "
      "terminal pose is invalid, the scene shows the preceding finite pose; "
      "the status panel reports the original nonfinite values."
    )
    viewer.run()
  finally:
    viewer.server.stop()


if __name__ == "__main__":
  tyro.cli(visualize, config=mjlab.TYRO_FLAGS)
