import argparse
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_URL = "https://raw.githubusercontent.com/PingoLH/Pytorch-HarDNet/master/hardnet68.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="weights/hardnet68.pth",
        help="Output .pth path.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="Override download URL.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "colonoscopy-segmentation-reproduction"})
    with urlopen(request) as response, open(destination, "wb") as output_file:
        shutil.copyfileobj(response, output_file)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url}")
    download_file(args.url, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
