"""Parse Insta360 INSV/INSP trailers (X4 Air directory-table layout)."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"8db42d694ccc418790edff439fe026bf"
RECORD_NAMES = {
    0x101: "maker_notes",
    0x2: "preview",
    0x3: "imu",
    0x4: "exposure",
    0x9: "unknown_0x09",
    0xA: "unknown_0x0a",
    0xB: "unknown_0x0b",
    0xC: "unknown_0x0c",
    0x16: "unknown_0x16",
    0x1B: "unknown_0x1b",
    0x1C: "unknown_0x1c",
    0x1D: "unknown_0x1d",
    0x200: "preview_legacy",
    0x300: "imu_legacy",
    0x400: "exposure_legacy",
}


@dataclass
class TrailerRecord:
    record_id: int
    size: int
    offset: int  # bytes from start of trailer
    payload: bytes

    @property
    def name(self) -> str:
        return RECORD_NAMES.get(self.record_id, f"unknown_{self.record_id:#x}")


def read_trailer(path: Path) -> tuple[int, dict[int, TrailerRecord]]:
    path = Path(path)
    with path.open("rb") as f:
        f.seek(-32, 2)
        magic = f.read(32)
        if magic != MAGIC:
            raise ValueError(f"{path} is not an Insta360 trailer (bad magic)")

        f.seek(-78, 2)
        footer = f.read(78)
        trailer_len = struct.unpack_from("<I", footer, 38)[0]
        first_id, first_len = struct.unpack_from("<HI", footer, 0)

        f.seek(-78 - first_len, 2)
        table = f.read(first_len)

        records: dict[int, TrailerRecord] = {}
        if first_id == 0 and first_len:
            for i in range(0, len(table) - 9, 10):
                rid, size, offset = struct.unpack_from("<HII", table, i)
                if rid == 0 or size == 0:
                    continue
                f.seek(-trailer_len + offset, 2)
                payload = f.read(size)
                records[rid] = TrailerRecord(rid, size, offset, payload)
        return trailer_len, records


def parse_maker_notes(payload: bytes) -> dict[str, str]:
    notes: dict[str, str] = {}
    i = 0
    while i + 2 <= len(payload):
        tag, n = payload[i], payload[i + 1]
        i += 2
        if n == 0 or i + n > len(payload):
            break
        raw = payload[i : i + n]
        i += n
        try:
            text = raw.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            text = raw.hex()
        if tag == 0x0A:
            notes["serial"] = text
        elif tag == 0x12:
            notes["model"] = text
        elif tag == 0x1A:
            notes["firmware"] = text
        elif tag == 0x2A:
            notes["offset_v3"] = text
        else:
            notes[f"tag_{tag:#x}"] = text
        if len(notes) > 16:
            break
    return notes


def parse_imu_20(payload: bytes):
    import numpy as np

    n = len(payload) // 20
    raw = np.frombuffer(payload[: n * 20], dtype="<u2").reshape(n, 10)
    ts_us = raw[:, :4].view("<u8").reshape(-1)
    xyz = (raw[:, 4:10].astype(np.float64) - 0x8000) / 1000.0
    t0 = ts_us[0]
    return {
        "t_s": (ts_us - t0) / 1e6,
        "ts_us": ts_us,
        "acc": xyz[:, :3],
        "gyro": xyz[:, 3:],
    }


def parse_exposure(payload: bytes):
    import numpy as np

    n = len(payload) // 16
    rec = np.frombuffer(payload[: n * 16], dtype=[("ts_us", "<u8"), ("exp_s", "<f8")])
    t0 = rec["ts_us"][0]
    return {
        "t_s": (rec["ts_us"] - t0) / 1e6,
        "ts_us": rec["ts_us"].copy(),
        "exp_s": rec["exp_s"].copy(),
    }


def dump_records_summary(records: dict[int, TrailerRecord]) -> list[dict]:
    rows = []
    for rid, rec in sorted(records.items()):
        rows.append(
            {
                "id": rid,
                "id_hex": hex(rid),
                "name": rec.name,
                "size": rec.size,
                "offset": rec.offset,
            }
        )
    return rows
