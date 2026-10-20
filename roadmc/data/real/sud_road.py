"""Evidence-bounded, memory-mapped reader for the supplied SUD-ROAD LAS 1.2 file.

The LAS Classification byte is preserved as a byte plus its low-five/high-three
bit components. This module deliberately does not translate it to SUD-ROAD
semantic classes or assign physical units.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

LAS_HEADER_MIN = 227
LAS_V12_HEADER = 227
FORMAT0_LENGTH = 20
_RECORD_DTYPE = np.dtype(
    [
        ("x_int", "<i4"),
        ("y_int", "<i4"),
        ("z_int", "<i4"),
        ("intensity", "<u2"),
        ("return_flags", "u1"),
        ("classification_raw", "u1"),
        ("scan_angle_rank", "i1"),
        ("user_data", "u1"),
        ("point_source_id", "<u2"),
    ],
    align=False,
)


class SUDLASFormatError(ValueError):
    """The file is outside the supported, internally consistent LAS subset."""


@dataclass(frozen=True)
class LASHeader:
    source_path: str
    file_size: int
    version: str
    header_size: int
    point_offset: int
    vlr_count: int
    point_format: int
    record_length: int
    legacy_point_count: int
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    bounds_xyz: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    system_id: str
    software_id: str
    global_encoding: int
    vlrs: tuple[dict[str, Any], ...]


def parse_header(path: str | Path) -> LASHeader:
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as f:
        h = f.read(LAS_HEADER_MIN)
    if len(h) != LAS_HEADER_MIN or h[:4] != b"LASF":
        raise SUDLASFormatError("missing/truncated LASF public header")
    major, minor = h[24], h[25]
    if (major, minor) != (1, 2):
        raise SUDLASFormatError(f"supported LAS version is 1.2, got {major}.{minor}")
    global_encoding = struct.unpack_from("<H", h, 6)[0]
    header_size, point_offset, vlr_count = struct.unpack_from("<HII", h, 94)
    fmt = h[104]
    record_length = struct.unpack_from("<H", h, 105)[0]
    count = struct.unpack_from("<I", h, 107)[0]
    scales = struct.unpack_from("<3d", h, 131)
    offsets = struct.unpack_from("<3d", h, 155)
    # LAS order: maxX,minX,maxY,minY,maxZ,minZ.
    mm = struct.unpack_from("<6d", h, 179)
    bounds = ((mm[1], mm[0]), (mm[3], mm[2]), (mm[5], mm[4]))
    if header_size < LAS_V12_HEADER or point_offset < header_size or point_offset > size:
        raise SUDLASFormatError("header size or point-data offset is inconsistent")
    if fmt != 0 or record_length != FORMAT0_LENGTH:
        raise SUDLASFormatError(f"supported point format is 0/20 bytes, got {fmt}/{record_length}")
    if not count:
        raise SUDLASFormatError("legacy point count is zero")
    end = point_offset + count * record_length
    if end != size:
        raise SUDLASFormatError(f"point-data boundary mismatch: expected {end}, actual {size}")
    if (
        not np.isfinite(scales).all()
        or any(v <= 0 for v in scales)
        or not np.isfinite(offsets).all()
    ):
        raise SUDLASFormatError("coordinate scales/offsets are invalid")
    if not np.isfinite(bounds).all() or any(lo > hi for lo, hi in bounds):
        raise SUDLASFormatError("header coordinate bounds are invalid")
    vlrs = []
    cursor = header_size
    with p.open("rb") as f:
        for index in range(vlr_count):
            if cursor + 54 > point_offset:
                raise SUDLASFormatError(f"VLR {index} header overlaps point data")
            f.seek(cursor)
            vh = f.read(54)
            if len(vh) != 54:
                raise SUDLASFormatError(f"VLR {index} header is truncated")
            reserved = struct.unpack_from("<H", vh, 0)[0]
            user_id = vh[2:18].split(b"\0", 1)[0].decode("ascii", "replace").strip()
            record_id, payload_length = struct.unpack_from("<HH", vh, 18)
            description = vh[22:54].split(b"\0", 1)[0].decode("ascii", "replace").strip()
            cursor += 54
            if cursor + payload_length > point_offset:
                raise SUDLASFormatError(f"VLR {index} payload overlaps point data")
            payload = f.read(payload_length)
            if len(payload) != payload_length:
                raise SUDLASFormatError(f"VLR {index} payload is truncated")
            item: dict[str, Any] = {
                "index": index,
                "reserved": reserved,
                "user_id": user_id,
                "record_id": record_id,
                "description": description,
                "payload_length": payload_length,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "payload_hex": payload.hex(),
            }
            if user_id == "LASF_Projection" and record_id == 34735:
                if payload_length % 2 or payload_length < 8:
                    raise SUDLASFormatError("GeoKeyDirectoryTag VLR has malformed SHORT payload")
                words = struct.unpack(f"<{payload_length // 2}H", payload)
                key_count = words[3]
                if len(words) != 4 + 4 * key_count:
                    raise SUDLASFormatError(
                        "GeoKeyDirectoryTag key count disagrees with its payload size"
                    )
                item["geokey_directory"] = {
                    "header": list(words[:4]),
                    "keys": [
                        {
                            "key_id": words[4 + i * 4],
                            "tiff_tag_location": words[5 + i * 4],
                            "count": words[6 + i * 4],
                            "value_offset": words[7 + i * 4],
                        }
                        for i in range(key_count)
                    ],
                }
            vlrs.append(item)
            cursor += payload_length
    if cursor > point_offset:
        raise SUDLASFormatError("VLR sequence exceeds point-data offset")
    return LASHeader(
        str(p.resolve()),
        size,
        f"{major}.{minor}",
        header_size,
        point_offset,
        vlr_count,
        fmt,
        record_length,
        count,
        tuple(scales),
        tuple(offsets),
        bounds,
        h[26:58].split(b"\0", 1)[0].decode("ascii", "replace").strip(),
        h[58:90].split(b"\0", 1)[0].decode("ascii", "replace").strip(),
        global_encoding,
        tuple(vlrs),
    )


def point_view(path: str | Path, header: LASHeader | None = None) -> np.ndarray:
    """Return a read-only structured memmap over raw point records."""
    p = Path(path)
    h = header or parse_header(p)
    if p.stat().st_size != h.file_size:
        raise SUDLASFormatError("source size changed after header validation")
    return np.memmap(
        p, dtype=_RECORD_DTYPE, mode="r", offset=h.point_offset, shape=(h.legacy_point_count,)
    )


def iter_blocks(
    points: np.ndarray, block_size: int = 1_000_000
) -> Iterator[tuple[int, np.ndarray]]:
    if isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    for start in range(0, len(points), block_size):
        yield start, points[start : min(start + block_size, len(points))]


def decoded_fields(records: np.ndarray, header: LASHeader) -> dict[str, np.ndarray]:
    """Decode coordinates and LAS 1.2 format-0 flag bits without relabeling."""
    c = records["classification_raw"]
    rf = records["return_flags"]
    ints = np.column_stack((records["x_int"], records["y_int"], records["z_int"]))
    xyz = ints.astype(np.float64) * np.asarray(header.scales) + np.asarray(header.offsets)
    return {
        "xyz_source_units": xyz,
        "xyz_integer_raw": ints.astype(np.int32, copy=False),
        "intensity_raw": np.asarray(records["intensity"], dtype=np.uint16),
        "return_flags_raw": np.asarray(rf, dtype=np.uint8),
        "classification_raw": np.asarray(c, dtype=np.uint8),
        "classification_low5": c & 0x1F,
        "classification_high3_flags": c >> 5,
        "return_number_raw": rf & 0x07,
        "number_of_returns_raw": (rf >> 3) & 0x07,
        "scan_direction_flag": (rf >> 6) & 1,
        "edge_of_flight_line": (rf >> 7) & 1,
        "scan_angle_rank_raw": np.asarray(records["scan_angle_rank"], dtype=np.int8),
        "user_data_raw": np.asarray(records["user_data"], dtype=np.uint8),
        "point_source_id_raw": np.asarray(records["point_source_id"], dtype=np.uint16),
    }


def sha256_file(path: str | Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def header_dict(header: LASHeader) -> dict:
    return asdict(header)
