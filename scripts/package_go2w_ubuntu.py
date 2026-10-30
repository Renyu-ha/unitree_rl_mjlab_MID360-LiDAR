"""Create a self-contained Ubuntu 22.04 x86_64 source release (stdlib only)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "go2w_ubuntu22_x86_64"


def collect():
  paths = set()
  trees = ["cmake", "patches", "deploy/include", "deploy/robots/go2w",
    "deploy/thirdparty/onnxruntime-linux-x64-1.22.0", "simulate/src", "simulate/mujoco",
    "simulate/robots/go2w", "unitree_sdk2/include", "unitree_sdk2/cmake",
    "unitree_sdk2/lib/x86_64", "unitree_sdk2/thirdparty/include", "unitree_sdk2/thirdparty/lib/x86_64",
    "src", "tests", "scripts", "doc"]
  for folder in trees:
    for path in (ROOT/folder).rglob("*"):
      rel = path.relative_to(ROOT)
      if not path.is_file() or any(part in {"__pycache__", "build", ".git", "motions"} for part in rel.parts): continue
      if "config/policy" in rel.as_posix(): continue
      if path.suffix.lower() in {".pyc", ".pyo", ".npz", ".pt", ".pth", ".mp4", ".exe", ".dll"}: continue
      if path.suffix == ".onnx" and not rel.as_posix().startswith("tests/fixtures/"): continue
      paths.add(path)
  for name in ["LICENCE", "README.md", "README_zh.md", "README_UBUNTU22_GO2W.md", "pyproject.toml", "environment.yml",
    "simulate/CMakeLists.txt", "simulate/config.yaml", "simulate/config_go2w.yaml",
    "unitree_sdk2/CMakeLists.txt", "unitree_sdk2/LICENSE", "unitree_sdk2/thirdparty/CMakeLists.txt"]:
    if (ROOT/name).is_file(): paths.add(ROOT/name)
  return sorted(paths)


def link_target(path):
  if path.is_symlink(): return path.readlink().as_posix()
  if ".so" in path.name and path.stat().st_size < 256:
    candidate = path.read_text().strip()
    if Path(candidate).name == candidate and (path.parent/candidate).is_file(): return candidate
  return None


def create(output):
  output = Path(output).resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  manifest = {"platform": "Ubuntu 22.04 x86_64", "files": {}}
  with tarfile.open(output, "w:gz", compresslevel=6) as archive:
    for path in collect():
      relative = path.relative_to(ROOT).as_posix()
      info = tarfile.TarInfo(PREFIX+"/"+relative)
      info.mode = 0o755 if path.suffix == ".sh" else 0o644
      target = link_target(path)
      if target:
        info.type = tarfile.SYMTYPE
        info.linkname = target
        manifest["files"][relative] = {"symlink": target}
        archive.addfile(info)
      else:
        data = path.read_bytes()
        info.size = len(data)
        manifest["files"][relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        archive.addfile(info, io.BytesIO(data))
    data = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    info = tarfile.TarInfo(PREFIX+"/RELEASE_MANIFEST.json")
    info.mode = 0o644; info.size = len(data)
    archive.addfile(info, io.BytesIO(data))
  digest = hashlib.sha256(output.read_bytes()).hexdigest()
  output.with_suffix(output.suffix+".sha256").write_text(f"{digest}  {output.name}\n",encoding="ascii",newline="\n")
  print(f"{output}: {len(manifest['files'])} files, {output.stat().st_size} bytes\nSHA256 {digest}")
  return manifest


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, default=ROOT/"outputs/go2w_ubuntu22_x86_64.tar.gz")
  create(parser.parse_args().output)
