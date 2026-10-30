"""Verify the unmodified source release before building (Python stdlib only)."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
  root = Path(root)
  manifest = json.loads((root/"RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
  for name, expected in manifest["files"].items():
    path = root/name
    if "symlink" in expected:
      if not path.is_symlink() or path.readlink().as_posix() != expected["symlink"]:
        raise RuntimeError(f"Incorrect shared library link: {name}")
    elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
      raise RuntimeError(f"Missing or modified release file: {name}")
  print(f"Verified {len(manifest['files'])} release files")


if __name__ == "__main__":
  verify()
