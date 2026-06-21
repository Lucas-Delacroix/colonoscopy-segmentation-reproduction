from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "upstream" / "commands.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Model keys selected for setup.")
    return parser.parse_args()


def env_name(env_file: Path) -> str:
    with open(env_file) as file:
        return yaml.safe_load(file)["name"]


def run(command: list[str], cwd: Path) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=clean_parent_python_env(), check=True)


def clean_parent_python_env() -> dict[str, str]:
    env = os.environ.copy()
    virtual_env = env.pop("VIRTUAL_ENV", None)
    if virtual_env:
        virtual_env_bin = str(Path(virtual_env) / "bin")
        path_parts = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(part for part in path_parts if part != virtual_env_bin)
    return env


def main() -> None:
    args = parse_args()
    selected = set(args.only) if args.only else None
    if selected is not None and "ssformer" not in selected:
        return

    with open(COMMANDS) as file:
        entry = yaml.safe_load(file)["commands"]["ssformer"]

    repo_dir = ROOT / entry["cwd"]
    env = env_name(ROOT / entry["env"])
    source = ROOT / "weights" / "swin_base_patch4_window7_224_22k.pth"
    target = repo_dir / "pretrain" / "swin_base_patch4_window7_224_22k.pth"
    converter = repo_dir / "tools" / "model_converters" / "swin2mmseg.py"

    if not repo_dir.exists():
        raise FileNotFoundError(f"SSFormer vendor repo not found: {repo_dir}")
    if not source.exists():
        raise FileNotFoundError(f"Swin checkpoint not found: {source}")
    if not converter.exists():
        raise FileNotFoundError(f"Swin converter not found: {converter}")

    run(
        [
            "conda",
            "run",
            "-n",
            env,
            "--no-capture-output",
            "python",
            os.path.relpath(converter, repo_dir),
            os.path.relpath(source, repo_dir),
            os.path.relpath(target, repo_dir),
        ],
        cwd=repo_dir,
    )


if __name__ == "__main__":
    main()
