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
    parser.add_argument("model", help="Model key from upstream/commands.yaml.")
    parser.add_argument("action", nargs="?", default="train", choices=("train", "export"))
    return parser.parse_args()


def load_commands() -> dict:
    with open(COMMANDS) as file:
        return yaml.safe_load(file)["commands"]


def conda_env_name(env_ref: str) -> str | None:
    if not env_ref.endswith((".yml", ".yaml")):
        return None
    with open(ROOT / env_ref) as file:
        return yaml.safe_load(file)["name"]


def clean_parent_python_env() -> dict[str, str]:
    env = os.environ.copy()
    virtual_env = env.pop("VIRTUAL_ENV", None)
    if virtual_env:
        virtual_env_bin = str(Path(virtual_env) / "bin")
        path_parts = env.get("PATH", "").split(os.pathsep)
        env["PATH"] = os.pathsep.join(part for part in path_parts if part != virtual_env_bin)
    return env


def run_command(entry: dict, action: str) -> None:
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
    subprocess.run(resolved, cwd=cwd, env=clean_parent_python_env(), check=True)


def main() -> None:
    args = parse_args()
    commands = load_commands()
    if args.model not in commands:
        raise SystemExit(f"Unknown model '{args.model}'. Options: {list(commands)}")
    entry = commands[args.model]
    if args.action not in entry:
        raise SystemExit(f"Action '{args.action}' is not configured for '{args.model}'.")
    run_command(entry, args.action)


if __name__ == "__main__":
    main()
