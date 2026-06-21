from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "upstream" / "commands.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model key from upstream/commands.yaml.")
    parser.add_argument("action", nargs="?", default="train", choices=("train", "export"))
    return parser.parse_args()


def load_commands() -> dict:
    with COMMANDS.open() as file:
        return yaml.safe_load(file)["commands"]


def conda_env_name(env_ref: str) -> str | None:
    if not env_ref.endswith((".yml", ".yaml")):
        return None
    with (ROOT / env_ref).open() as file:
        return yaml.safe_load(file)["name"]


def clean_parent_python_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    virtual_env = env.pop("VIRTUAL_ENV", None)
    if virtual_env:
        virtual_env_bin = str(Path(virtual_env) / "bin")
        path_parts = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(part for part in path_parts if part != virtual_env_bin)
    return env


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def run_command(model: str, entry: dict, action: str) -> None:
    cwd = ROOT / entry.get(f"{action}_cwd", entry["cwd"])
    command = entry[action]
    env_name = conda_env_name(entry["env"])
    command_args = shlex.split(command)

    if env_name:
        resolved = ["conda", "run", "-n", env_name, "--no-capture-output", *command_args]
    else:
        resolved = command_args

    print(f"cwd: {cwd}")
    print("+ " + shlex.join(resolved), flush=True)
    print(f"[{timestamp()}] START {model} {action}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(resolved, cwd=cwd, env=clean_parent_python_env(), check=False)
    duration = format_duration(time.monotonic() - started)
    if completed.returncode != 0:
        print(
            f"[{timestamp()}] END {model} {action}: FAILED "
            f"exit={completed.returncode}; duration={duration}",
            flush=True,
        )
        raise SystemExit(completed.returncode)
    print(f"[{timestamp()}] END {model} {action}: OK; duration={duration}", flush=True)


def main() -> None:
    args = parse_args()
    commands = load_commands()
    if args.model not in commands:
        raise SystemExit(f"Unknown model '{args.model}'. Options: {list(commands)}")
    entry = commands[args.model]
    if args.action not in entry:
        raise SystemExit(f"Action '{args.action}' is not configured for '{args.model}'.")
    run_command(args.model, entry, args.action)


if __name__ == "__main__":
    main()
