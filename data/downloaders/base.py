import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.request import urlretrieve


class DatasetDownloader(ABC):
    name: str
    url: str
    archive_name: str
    dataset_dir_name: str | None = None
    sha256: str | None = None

    def __init__(
        self,
        raw_root: str | Path = "data/raw",
        downloads_root: str | Path = "data/downloads",
        force: bool = False,
    ):
        self.raw_root = Path(raw_root)
        self.downloads_root = Path(downloads_root)
        self.force = force

    @property
    def dataset_dir(self) -> Path:
        return self.raw_root / (self.dataset_dir_name or self.name)

    @property
    def archive_path(self) -> Path:
        return self.downloads_root / self.archive_name

    def run(self) -> Path:
        """Download, prepare and validate the dataset."""
        if self.is_available() and not self.force:
            self.validate()
            print(f"Dataset already available: {self.dataset_dir}")
            return self.dataset_dir

        self.download()
        self.prepare()
        self.validate()

        return self.dataset_dir

    def download(self) -> None:
        """Download the raw archive if needed."""
        self.downloads_root.mkdir(parents=True, exist_ok=True)

        if self.archive_path.exists() and not self.force:
            self._verify_archive()
            print(f"Archive already downloaded: {self.archive_path}")
            return

        print(f"Downloading {self.name} from {self.url}")
        partial_path = self.archive_path.with_suffix(self.archive_path.suffix + ".part")
        urlretrieve(self.url, partial_path)
        partial_path.replace(self.archive_path)
        self._verify_archive()
        print(f"Saved archive: {self.archive_path}")

    def _verify_archive(self) -> None:
        if self.sha256 is None:
            return

        digest = hashlib.sha256()
        with open(self.archive_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)

        actual = digest.hexdigest()
        if actual != self.sha256:
            raise ValueError(
                f"Checksum mismatch for {self.archive_path}. "
                f"Expected {self.sha256}, got {actual}."
            )

    @abstractmethod
    def prepare(self) -> None:
        """Extract and normalize the dataset into dataset_dir."""
        ...

    @abstractmethod
    def validate(self) -> None:
        """Raise a clear error if the prepared dataset is invalid."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the prepared dataset already exists locally."""
        ...
