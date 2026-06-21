from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "upstream" / "envs"
COMMANDS = ROOT / "upstream" / "commands.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Model keys or environment files to create. Defaults to every upstream/envs/*.yml file.",
    )
    parser.add_argument(
        "--recreate-existing",
        action="store_true",
        help="Remove and create environments that already exist.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Run conda env update --prune for environments that already exist.",
    )
    parser.add_argument(
        "--solver",
        choices=("classic", "libmamba"),
        default=None,
        help="Optional Conda solver backend for create/update/remove commands.",
    )
    args = parser.parse_args()
    if args.recreate_existing and args.update_existing:
        parser.error("--recreate-existing and --update-existing cannot be used together.")
    return args


def env_name(env_file: Path) -> str:
    with open(env_file) as file:
        data = yaml.safe_load(file)
    return data["name"]


def existing_env_names() -> set[str]:
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    env_paths = json.loads(result.stdout).get("envs", [])
    return {Path(env_path).name for env_path in env_paths}


def load_commands() -> dict:
    with open(COMMANDS) as file:
        return yaml.safe_load(file)["commands"]


def selected_env_files(only: list[str] | None) -> list[Path]:
    if not only:
        return sorted(ENV_DIR.glob("*.yml"))

    commands = load_commands()
    env_files: list[Path] = []
    seen: set[Path] = set()
    for item in only:
        if item in commands:
            env_file = ROOT / commands[item]["env"]
        else:
            candidate = Path(item)
            if candidate.suffix not in {".yml", ".yaml"}:
                candidate = ENV_DIR / f"{item}.yml"
            env_file = candidate if candidate.is_absolute() else ROOT / candidate

        env_file = env_file.resolve()
        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found for '{item}': {env_file}")
        if env_file not in seen:
            env_files.append(env_file)
            seen.add(env_file)
    return env_files


def run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def solver_args(solver: str | None) -> list[str]:
    return ["--solver", solver] if solver else []


def main() -> None:
    args = parse_args()
    solver = solver_args(args.solver)
    existing = existing_env_names()
    for env_file in selected_env_files(args.only):
        name = env_name(env_file)
        if name in existing:
            if args.recreate_existing:
                run(["conda", "env", "remove", "-n", name, "-y", *solver])
                run(["conda", "env", "create", "-f", str(env_file), "-y", *solver])
            elif args.update_existing:
                run(["conda", "env", "update", "-n", name, "-f", str(env_file), "--prune", *solver])
            else:
                print(
                    f"Already exists: {name} ({env_file}); skipping. "
                    "Use RECREATE_ENVS=1 to remove and create it again."
                )
        else:
            run(["conda", "env", "create", "-f", str(env_file), "-y", *solver])


if __name__ == "__main__":
    main()
