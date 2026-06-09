import argparse
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_URL = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="weights/swin_base_patch4_window7_224_22k.pth",
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
