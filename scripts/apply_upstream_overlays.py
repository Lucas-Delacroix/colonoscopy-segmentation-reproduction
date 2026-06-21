from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "upstream" / "commands.yaml"
VENDOR = ROOT / "upstream" / "vendor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Model keys to apply overlays for.")
    return parser.parse_args()


def apply_replacements(repo_dir: Path, name: str, replacements: list[dict]) -> None:
    for replacement in replacements:
        target = repo_dir / replacement["target"]
        old = replacement["old"]
        new = replacement["new"]
        text = target.read_text()
        if old not in text:
            if new in text:
                print(f"{name}: already patched {target}")
                continue
            raise ValueError(f"Could not find replacement text in {target}: {old!r}")
        target.write_text(text.replace(old, new, 1))
        print(f"{name}: patched {target}")


def main() -> None:
    args = parse_args()
    selected = set(args.only) if args.only else None
    with open(COMMANDS) as file:
        commands = yaml.safe_load(file)["commands"]

    for name, entry in commands.items():
        if selected is not None and name not in selected:
            continue
        overlays = entry.get("overlay", [])
        replacements = entry.get("replace", [])
        if not overlays and not replacements:
            continue
        repo_dir = ROOT / entry["cwd"]
        if not repo_dir.exists():
            raise FileNotFoundError(f"Vendor repo not found: {repo_dir}")
        for overlay in overlays:
            source = (repo_dir / overlay["source"]).resolve()
            target = repo_dir / overlay["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"{name}: {source} -> {target}")
        apply_replacements(repo_dir, name, replacements)


if __name__ == "__main__":
    main()
