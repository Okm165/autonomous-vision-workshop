#!/usr/bin/env python3
"""Fetch the KITTI odometry sequence 00 excerpt used by this workshop.

The KITTI odometry benchmark distributes both cameras for all 22 sequences in
``data_odometry_gray.zip`` (~23 GB), served from the official AWS S3 mirror
(``s3.eu-central-1.amazonaws.com/avg-kitti``).  Registration is required only
for the *ground-truth* archives; the image archive is public and, because S3
serves byte ranges, the few files this workshop needs can be extracted directly
out of the remote central directory without downloading the archive.

This script reads the ZIP central directory over HTTP, locates the entries for
sequence 00, and issues one ranged request per entry:

* ``dataset/sequences/00/calib.txt``  -> ``data/kitti/sequences/00/calib.txt``
* ``dataset/sequences/00/times.txt``  -> ``data/kitti/sequences/00/times.txt``
* ``dataset/sequences/00/image_0/<frame>.png`` -> left (grayscale) frames
* ``dataset/sequences/00/image_1/<frame>.png`` -> right (grayscale) frames

About 12 MB of the 23 GB archive is transferred.  The script is idempotent:
frames already on disk are skipped, so re-running it resumes an interrupted
fetch.

Usage
-----
    python data/fetch_kitti_frames.py [--frames 200] [--jobs 8]
"""

from __future__ import annotations

import argparse
import struct
import sys
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"
ZIP_LOCAL_HEADER = 0x04034B50
# Fixed size of a local file header, up to (not including) the file name:
#  sig(4) ver(2) flags(2) method(2) mtime(2) mdate(2) crc(4)
#  csize(4) usize(4) name_len(2) extra_len(2)
LOCAL_HEADER_SIZE = 30
ZIP_CENTRAL_HEADER = 0x02014B50
ZIP_EOCD = 0x06054B50
ZIP64_EOCD = 0x06064B50
ZIP64_LOCATOR = 0x07064B50
# The archive is 23 GB, so it is ZIP64: the trailing EOCD stores 0xFFFFFFFF for
# both the central-directory offset and the entry count, and the real values live
# in a ZIP64 EOCD record pointed at by a locator.  ``data_odometry_gray.zip`` is
# the worst case -- it holds more than 65535 entries, so its 16-bit entry count
# is saturated too -- which is why the ZIP64 record must always be consulted.
ZIP64_OFFSET_SENTINEL = 0xFFFFFFFF
# EOCD is followed by an archive comment of at most 65535 bytes, but reading a
# fixed 64 KiB tail keeps the request small while always containing the EOCD.
EOCD_SEARCH_WINDOW = 1 << 16
CENTRAL_HEADER_FIXED = 46
ZIP64_EXTRA_ID = 0x0001
# Central-directory field offsets, all relative to the record start:
#   0 signature(4)   4 version made by(2)  6 version needed(2)  8 flags(2)
#  10 method(2)     12 mod time(2)        14 mod date(2)        16 crc(4)
#  20 compressed size(4)                  24 uncompressed size(4)
#  28 name length(2) 30 extra length(2)   32 comment length(2)
#  34 disk start(2)  36 internal attrs(2) 38 external attrs(4)  42 local offset(4)
#  46 name
_CD_METHOD = 10
_CD_CSIZE = 20
_CD_LOCAL_OFFSET = 42
USER_AGENT = "autonomus-workshop-kitti-fetch/1.0"


def _fetch_range(url: str, start: int, end: int) -> bytes:
    """Return bytes ``[start, end]`` inclusive from ``url`` via a ranged GET."""
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise urllib.error.URLError(
                f"server ignored Range header (HTTP {response.status}); "
                "a ranged GET is required to avoid the 23 GB download"
            )
        return response.read()


def _read_central_header(url: str) -> tuple[int, int, int]:
    """Return ``(entry_count, cd_offset, cd_size)`` for the remote archive.

    Reads the trailing records only: the ZIP64 EOCD when present (which this
    23 GB archive always has), otherwise the legacy 32-bit EOCD.
    """
    head = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(head, timeout=60) as response:
        size = int(response.headers["Content-Length"])

    tail_start = max(0, size - EOCD_SEARCH_WINDOW)
    tail = _fetch_range(url, tail_start, size - 1)

    locator_at = tail.rfind(struct.pack("<I", ZIP64_LOCATOR))
    if locator_at >= 0:
        # Zip64EndOfCentralDirectoryLocator: sig, disk, eocd_offset(8), disks
        (zip64_eocd_offset,) = struct.unpack_from("<Q", tail, locator_at + 8)
        record = _fetch_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
        (signature,) = struct.unpack_from("<I", record, 0)
        if signature != ZIP64_EOCD:
            raise RuntimeError(
                f"ZIP64 locator points at offset {zip64_eocd_offset}, "
                "which does not hold a ZIP64 end-of-central-directory record"
            )
        entry_count = struct.unpack_from("<Q", record, 32)[0]
        cd_size, cd_offset = struct.unpack_from("<QQ", record, 40)
        return entry_count, cd_offset, cd_size

    eocd_at = tail.rfind(struct.pack("<I", ZIP_EOCD))
    if eocd_at < 0:
        raise RuntimeError("end-of-central-directory record not found")
    entry_count = struct.unpack_from("<H", tail, eocd_at + 10)[0]
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd_at + 12)
    return entry_count, cd_offset, cd_size


def _parse_entries(cd: bytes) -> dict[str, tuple[int, int, int, int]]:
    """Parse the central directory into ``name -> (method, usize, offset, csize)``.

    Saturated 32-bit sizes and offsets are resolved through the ZIP64 extra
    field; entries no larger than 4 GB simply use their 32-bit values.
    """
    entries: dict[str, tuple[int, int, int, int]] = {}
    pos = 0
    while pos + CENTRAL_HEADER_FIXED <= len(cd):
        (signature,) = struct.unpack_from("<I", cd, pos)
        if signature != ZIP_CENTRAL_HEADER:
            break
        (method,) = struct.unpack_from("<H", cd, pos + _CD_METHOD)
        csize, usize, name_len, extra_len, comment_len = struct.unpack_from(
            "<IIHHH", cd, pos + _CD_CSIZE
        )
        (local_offset,) = struct.unpack_from("<I", cd, pos + _CD_LOCAL_OFFSET)
        name_at = pos + CENTRAL_HEADER_FIXED
        name = cd[name_at : name_at + name_len].decode("utf-8", errors="replace")
        extra_at = name_at + name_len
        if (
            usize == ZIP64_OFFSET_SENTINEL
            or csize == ZIP64_OFFSET_SENTINEL
            or local_offset == ZIP64_OFFSET_SENTINEL
        ):
            # The ZIP64 field stores only the saturated values, in the order
            # uncompressed size, compressed size, local offset.
            extra = cd[extra_at : extra_at + extra_len]
            values = _zip64_field(extra)
            index = 0
            if usize == ZIP64_OFFSET_SENTINEL:
                usize = values[index]
                index += 1
            if csize == ZIP64_OFFSET_SENTINEL:
                csize = values[index]
                index += 1
            if local_offset == ZIP64_OFFSET_SENTINEL:
                local_offset = values[index]
        entries[name] = (method, usize, local_offset, csize)
        pos = extra_at + extra_len + comment_len
    return entries


def _zip64_field(extra: bytes) -> tuple[int, ...]:
    """Return the integer values of the ZIP64 extra field."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        if header_id == ZIP64_EXTRA_ID:
            return struct.unpack_from(f"<{data_size // 8}Q", extra, pos + 4)
        pos += 4 + data_size
    raise RuntimeError("ZIP64 extra field missing for a saturated central entry")


def _extract_entry(
    url: str,
    method: int,
    usize: int,
    local_offset: int,
    csize: int,
) -> bytes:
    """Download and, when stored compressed, inflate a single archive entry.

    The local header is the authoritative source for where the file data starts:
    it repeats the name and extra-field lengths, which the central directory does
    not record.  The 32-bit size slots in a local header are saturated for ZIP64
    entries, but the *lengths* still hold, so no extra-field parsing is needed.
    """
    fixed = _fetch_range(url, local_offset, local_offset + LOCAL_HEADER_SIZE - 1)
    (signature,) = struct.unpack_from("<I", fixed, 0)
    if signature != ZIP_LOCAL_HEADER:
        raise RuntimeError(f"bad local header at offset {local_offset}")
    name_len, extra_len = struct.unpack_from("<HH", fixed, 26)
    data_at = local_offset + LOCAL_HEADER_SIZE + name_len + extra_len
    raw = _fetch_range(url, data_at, data_at + csize - 1)
    if method == 0:
        return raw
    if method == 8:
        return zlib.decompress(raw, -zlib.MAX_WBITS)
    raise RuntimeError(f"unsupported compression method {method}")


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temporary file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="number of frames to fetch per camera (default: 200)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="parallel downloads (default: 8)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="destination data/ directory (default: this script's directory)",
    )
    args = parser.parse_args(argv)

    seq_dir = args.data_root / "kitti" / "sequences" / "00"
    print(f"Reading central directory of {ARCHIVE_URL}")
    entry_count, cd_offset, cd_size = _read_central_header(ARCHIVE_URL)
    print(
        f"  {entry_count} entries, central directory {cd_size / 1e6:.1f} MB at {cd_offset}"
    )
    entries = _parse_entries(
        _fetch_range(ARCHIVE_URL, cd_offset, cd_offset + cd_size - 1)
    )
    print(f"  parsed {len(entries)} entry names")

    prefix = "dataset/sequences/00/"
    tasks: list[tuple[str, Path, tuple[int, int, int, int]]] = []
    for name, meta in entries.items():
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix) :]
        if relative in {"calib.txt", "times.txt"}:
            tasks.append((name, seq_dir / relative, meta))
            continue
        camera, _, filename = relative.partition("/")
        if camera not in {"image_0", "image_1"} or not filename.endswith(".png"):
            continue
        if int(Path(filename).stem) >= args.frames:
            continue
        tasks.append((name, seq_dir / camera / filename, meta))

    missing = [
        (name, dest, meta)
        for name, dest, meta in tasks
        if not dest.is_file() or dest.stat().st_size != meta[1]
    ]
    print(f"  {len(tasks)} needed, {len(tasks) - len(missing)} already present")

    if not missing:
        print(f"Already complete: {seq_dir}")
        return 0

    failed: list[str] = []
    done = 0

    def download(
        item: tuple[str, Path, tuple[int, int, int, int]],
    ) -> str | None:
        name, dest, (method, usize, offset, csize) = item
        try:
            payload = _extract_entry(ARCHIVE_URL, method, usize, offset, csize)
            if len(payload) != usize:
                return f"{name}: got {len(payload)} bytes, expected {usize}"
            _write_atomic(dest, payload)
        except (urllib.error.URLError, OSError, RuntimeError, struct.error) as exc:
            return f"{name}: {exc}"
        return None

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for error in pool.map(download, missing):
            done += 1
            if error is not None:
                failed.append(error)
            if done % 50 == 0 or done == len(missing):
                print(f"  {done}/{len(missing)} fetched")

    if failed:
        print(f"\n{len(failed)} entr(ies) failed:", file=sys.stderr)
        for error in failed[:10]:
            print(f"  {error}", file=sys.stderr)
        return 1

    left = len(list((seq_dir / "image_0").glob("*.png")))
    right = len(list((seq_dir / "image_1").glob("*.png")))
    print(
        f"\nDone: image_0={left} frames, image_1={right} frames, plus calib.txt/times.txt"
    )
    print("Ground-truth poses are fetched by data/download.sh separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
