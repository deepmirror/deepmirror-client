"""deepmirror utils

This module provides utility functions for the deepmirror API client.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from httpx import Response
from tqdm import tqdm


@dataclass
class ProgressFile:
    """Wraps a file-like object to report upload progress via callback."""

    _file: BinaryIO
    _callback: Callable[[int, int], None]
    _size: int = field(init=False)
    _read: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._size = self._get_size()

    def _get_size(self) -> int:
        try:
            return self._file.seek(0, 2) or 0
        finally:
            self._file.seek(0)

    def read(self, amt: int = -1) -> bytes:
        """Read a chunk of data from the file and report progress."""
        chunk = self._file.read(amt)
        if chunk:
            self._read += len(chunk)
            self._callback(self._read, self._size)
        return chunk

    def tell(self) -> int:
        """Expose file's current position (satisfies pylint public method count)."""
        return self._file.tell()


def create_upload_files(
    file_path: str, file_obj: BinaryIO
) -> tuple[str, ProgressFile, str]:
    """Create a monitored upload file tuple for use in httpx `files=`."""
    filename = Path(file_path).name
    size = file_obj.seek(0, 2)
    file_obj.seek(0)
    callback = _create_callback(size)
    return (
        filename,
        ProgressFile(file_obj, callback),
        "application/octet-stream",
    )


def _create_callback(total_size: int) -> Callable[[int, int], None]:
    """Create a tqdm progress callback."""
    upload_bar = tqdm(
        total=total_size, unit="B", unit_scale=True, desc="Uploading"
    )

    def callback(read_bytes: int, _: int) -> None:
        upload_bar.update(read_bytes - upload_bar.n)
        if read_bytes >= total_size:
            upload_bar.close()

    return callback


def download_stream(response: Response) -> bytes:
    """Download a stream of data from a response."""
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        total=total_size, unit="iB", unit_scale=True, desc="Downloading results"
    )
    chunks = []
    for chunk in response.iter_bytes():
        if chunk:
            chunks.append(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()
    return b"".join(chunks)
