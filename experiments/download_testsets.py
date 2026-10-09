"""Fetch the optional Pascal Pons benchmark inputs and verify their recorded hashes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from experiments.benchmark import TESTSETS

BASE_URL = "http://blog.gamesolver.org/data/"
MAX_BYTES = 128 * 1024
SHA256 = {
    "Test_L1_R1": "003b7f5cc2b9b2d07ee2663cf92dfe99a50d7f4f57c556b6f35048e1bfaedb68",
    "Test_L1_R2": "ba3df3282804270661239d0fc357761a5c7cc31bbd581f3c502222abffa31113",
    "Test_L1_R3": "7b6eb6c7c9e7342f8065188cac01a25df2413a5b77de062cf14a4a8b237b140a",
    "Test_L2_R1": "52b9ee96ab6e92fd755ca4c545792c07c548a5a994fd32beec2775939d071b3c",
    "Test_L2_R2": "a0bb9983650ad5b6c6080da47c31065f6efa11bfadf2e1149af1619d7d65b745",
    "Test_L3_R1": "fae47639d993cc91f074d0b642a5f2bb251d31b15cea9df496d672c01fb2efec",
}


def download_file(name: str, directory: Path = TESTSETS) -> Path:
    expected = SHA256[name]
    target = directory / name
    exists = target.exists()
    if exists:
        payload = target.read_bytes()
    else:
        with urlopen(BASE_URL + name, timeout=30) as response:
            payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError(f"{name}: response exceeds the benchmark size limit.")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(
            f"{name}: SHA-256 mismatch; no file was written. "
            "Check the source or remove a damaged local copy before retrying."
        )
    if not exists:
        directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".download-", dir=directory) as staging:
            temporary = Path(staging) / name
            temporary.write_bytes(payload)
            temporary.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TESTSETS)
    args = parser.parse_args()
    try:
        for name in SHA256:
            downloaded = download_file(name, args.output)
            print(f"Verified {downloaded.name}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Benchmark download failed: {error}\n")


if __name__ == "__main__":
    main()