from __future__ import annotations

import argparse
import codecs
import os
import selectors
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMMANDS = ROOT / "upstream" / "commands.yaml"
DEFAULT_MODELS = [
    "hardnet_mseg",
    "hardnet_dfus",
    "ssformer",
    "tganet",
    "colonformer",
    "esfpnet",
    "meta_polyp",
    "cascade",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all upstream models sequentially for one or more full rounds.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys from upstream/commands.yaml. Defaults to the full table order.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of full sequential training rounds to execute.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next model if a training command fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the model order without starting training.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/logs",
        help="Directory where logs will be written.",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=60,
        help="Seconds between status messages while a command is still running. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Do not reapply upstream overlays before each model training command.",
    )
    return parser.parse_args()


def load_model_keys() -> set[str]:
    with open(COMMANDS) as file:
        return set(yaml.safe_load(file)["commands"])


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.status_interval < 0:
        raise SystemExit("--status-interval must be >= 0")

    known = load_model_keys()
    unknown = [model for model in args.models if model not in known]
    if unknown:
        raise SystemExit(
            f"Unknown model(s): {', '.join(unknown)}. Options: {', '.join(sorted(known))}"
        )


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def append_tail(tail: deque[str], buffer: str, text: str) -> str:
    buffer += text
    parts = buffer.splitlines(keepends=True)
    if parts and not parts[-1].endswith(("\n", "\r")):
        buffer = parts.pop()
    else:
        buffer = ""

    for part in parts:
        line = part.strip()
        if line:
            tail.append(line)
    return buffer


def write_all(text: str, logs: list) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()
    for log in logs:
        log.write(text)
        log.flush()


def run_and_log(
    command: list[str],
    combined_log: Path,
    model_log: Path,
    label: str,
    status_interval: int,
) -> tuple[int, float, list[str]]:
    started = time.monotonic()
    last_output = started
    next_status = started + status_interval if status_interval else float("inf")
    tail: deque[str] = deque(maxlen=20)
    tail_buffer = ""

    model_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with combined_log.open("a", encoding="utf-8") as combined, model_log.open(
        "a", encoding="utf-8"
    ) as model_file:
        header = (
            f"\n[{timestamp()}] START {label}\n"
            f"+ {shlex.join(command)}\n"
            f"model log: {model_log}\n"
        )
        write_all(header, [combined, model_file])

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=env,
        )
        assert process.stdout is not None

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        os.set_blocking(process.stdout.fileno(), False)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        stream_open = True

        while stream_open or process.poll() is None:
            for key, _ in selector.select(timeout=1):
                try:
                    data = os.read(key.fileobj.fileno(), 8192)
                except BlockingIOError:
                    continue

                if not data:
                    selector.unregister(key.fileobj)
                    stream_open = False
                    break

                text = decoder.decode(data)
                if text:
                    write_all(text, [combined, model_file])
                    last_output = time.monotonic()
                    tail_buffer = append_tail(tail, tail_buffer, text)

            now = time.monotonic()
            if now >= next_status:
                status = (
                    f"\n[{timestamp()}] STATUS {label}: running for "
                    f"{format_duration(now - started)}; last output "
                    f"{format_duration(now - last_output)} ago; log: {model_log}\n"
                )
                write_all(status, [combined, model_file])
                next_status = now + status_interval

        remainder = decoder.decode(b"", final=True)
        if remainder:
            write_all(remainder, [combined, model_file])
            tail_buffer = append_tail(tail, tail_buffer, remainder)
        if tail_buffer.strip():
            tail.append(tail_buffer.strip())

        return_code = process.wait()
        duration = time.monotonic() - started
        footer = (
            f"\n[{timestamp()}] END {label}: exit={return_code}; "
            f"duration={format_duration(duration)}\n"
        )
        write_all(footer, [combined, model_file])
        return return_code, duration, list(tail)


def main() -> None:
    args = parse_args()
    validate_args(args)

    log_root = ROOT / args.log_dir
    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = log_root / f"train_all_{started_at}"
    combined_log = run_log_dir / "combined.log"

    print("Training order:")
    for index, model in enumerate(args.models, start=1):
        print(f"  {index}. {model}")
    print(f"Log directory: {run_log_dir}")

    if args.dry_run:
        print("Dry run only. No training was started.")
        return

    run_log_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, str, str, int, float, Path]] = []
    failures: list[tuple[int, str, str, int, list[str], Path]] = []
    total_models = len(args.models)
    for round_number in range(1, args.rounds + 1):
        print(f"\n=== Training round {round_number}/{args.rounds} ===")
        for model_index, model in enumerate(args.models, start=1):
            model_log = run_log_dir / f"round_{round_number:02d}_{model}.log"
            print(f"\n=== Model {model_index}/{total_models}: {model} ===")

            if not args.skip_overlays:
                overlay_command = [
                    sys.executable,
                    "-m",
                    "scripts.apply_upstream_overlays",
                    "--only",
                    model,
                ]
                label = f"round {round_number}/{args.rounds} {model} overlays"
                return_code, duration, tail = run_and_log(
                    overlay_command,
                    combined_log,
                    model_log,
                    label,
                    args.status_interval,
                )
                results.append((round_number, model, "overlays", return_code, duration, model_log))
                if return_code != 0:
                    failures.append((round_number, model, "overlays", return_code, tail, model_log))
                    print(
                        f"Overlay failed for {model} in round {round_number} "
                        f"with exit code {return_code}."
                    )
                    if not args.continue_on_error:
                        print(f"Combined log: {combined_log}")
                        raise SystemExit(return_code)
                    continue

            command = [sys.executable, "-m", "scripts.run_upstream", model, "train"]
            label = f"round {round_number}/{args.rounds} {model} training"
            return_code, duration, tail = run_and_log(
                command,
                combined_log,
                model_log,
                label,
                args.status_interval,
            )
            results.append((round_number, model, "training", return_code, duration, model_log))
            if return_code != 0:
                failures.append((round_number, model, "training", return_code, tail, model_log))
                print(
                    f"Training failed for {model} in round {round_number} "
                    f"with exit code {return_code}."
                )
                if not args.continue_on_error:
                    print(f"Combined log: {combined_log}")
                    raise SystemExit(return_code)

    print(f"\nCombined log: {combined_log}")
    print("Summary:")
    for round_number, model, step, return_code, duration, model_log in results:
        status = "OK" if return_code == 0 else "FAIL"
        print(
            f"  {status} round {round_number} {model} {step}: "
            f"{format_duration(duration)} ({model_log})"
        )

    if failures:
        print("Failures:")
        for round_number, model, step, return_code, tail, model_log in failures:
            print(f"  round {round_number}: {model} {step} exited with {return_code}")
            print(f"  log: {model_log}")
            if tail:
                print("  last output:")
                for line in tail[-5:]:
                    print(f"    {line}")
        raise SystemExit(1)

    print("All training commands completed.")


if __name__ == "__main__":
    main()
