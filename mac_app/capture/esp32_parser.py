"""ESP32 CSI frame parser (RuView ADR-018 binary format).

Reference: RuView ``wifi-densepose-hardware/src/esp32_parser.rs``
"""
from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

ESP32_CSI_MAGIC = 0xC5110001
RUVIEW_VITALS_MAGIC = 0xC5110002
RUVIEW_FEATURE_MAGIC = 0xC5110003
RUVIEW_FUSED_VITALS_MAGIC = 0xC5110004
RUVIEW_COMPRESSED_CSI_MAGIC = 0xC5110005
RUVIEW_FEATURE_STATE_MAGIC = 0xC5110006
RUVIEW_TEMPORAL_MAGIC = 0xC5110007

HEADER_SIZE = 20
MAX_SUBCARRIERS = 256
MAX_ANTENNAS = 4

# Known fixed sizes for sibling packets (skip in stream resync).
SIBLING_PACKET_SIZES = {
    RUVIEW_VITALS_MAGIC: 32,
}

_HEADER_STRUCT = struct.Struct("<IBBHIIbbH")


class ParseErrorKind(str, Enum):
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_MAGIC = "invalid_magic"
    NON_CSI_PACKET = "non_csi_packet"
    INVALID_ANTENNA_COUNT = "invalid_antenna_count"
    INVALID_SUBCARRIER_COUNT = "invalid_subcarrier_count"


@dataclass
class ParseError(Exception):
    kind: ParseErrorKind
    message: str = ""
    magic: int = 0
    kind_name: str = ""

    def __str__(self) -> str:
        return self.message or self.kind.value


@dataclass
class Esp32CsiFrame:
    ts: float
    node_id: int
    n_antennas: int
    n_subcarriers: int
    channel_freq_mhz: int
    sequence: int
    rssi_dbm: int
    noise_floor_dbm: int
    """Per-subcarrier amplitude, shape (n_subcarriers,), antenna-averaged."""
    amplitude: np.ndarray
    source_ip: str = ""

    @property
    def subcarrier_count(self) -> int:
        return int(self.amplitude.shape[0])


def ruview_sibling_packet_name(magic: int) -> Optional[str]:
    names = {
        RUVIEW_VITALS_MAGIC: "ADR-039 edge vitals",
        RUVIEW_FEATURE_MAGIC: "ADR-069 feature vector",
        RUVIEW_FUSED_VITALS_MAGIC: "ADR-063 fused vitals",
        RUVIEW_COMPRESSED_CSI_MAGIC: "ADR-039 compressed CSI",
        RUVIEW_FEATURE_STATE_MAGIC: "ADR-081 feature state",
        RUVIEW_TEMPORAL_MAGIC: "ADR-095 temporal classification",
    }
    return names.get(magic)


def _read_magic(data: bytes) -> int:
    if len(data) < 4:
        return 0
    return struct.unpack_from("<I", data, 0)[0]


def parse_frame(data: bytes, *, source_ip: str = "", ts: Optional[float] = None) -> Tuple[Esp32CsiFrame, int]:
    if len(data) >= 4:
        magic = _read_magic(data)
        if name := ruview_sibling_packet_name(magic):
            raise ParseError(
                ParseErrorKind.NON_CSI_PACKET,
                message=f"non-CSI RuView packet: {name}",
                magic=magic,
                kind_name=name,
            )

    if len(data) < HEADER_SIZE:
        raise ParseError(
            ParseErrorKind.INSUFFICIENT_DATA,
            message=f"need {HEADER_SIZE} header bytes, got {len(data)}",
        )

    (
        magic,
        node_id,
        n_antennas,
        n_subcarriers,
        channel_freq_mhz,
        sequence,
        rssi_dbm,
        noise_floor_dbm,
        _reserved,
    ) = _HEADER_STRUCT.unpack_from(data, 0)

    if magic != ESP32_CSI_MAGIC:
        raise ParseError(
            ParseErrorKind.INVALID_MAGIC,
            message=f"invalid magic {magic:#010x}",
            magic=magic,
        )

    if n_antennas == 0 or n_antennas > MAX_ANTENNAS:
        raise ParseError(
            ParseErrorKind.INVALID_ANTENNA_COUNT,
            message=f"invalid antenna count {n_antennas}",
        )

    if n_subcarriers == 0 or n_subcarriers > MAX_SUBCARRIERS:
        raise ParseError(
            ParseErrorKind.INVALID_SUBCARRIER_COUNT,
            message=f"invalid subcarrier count {n_subcarriers}",
        )

    iq_pairs = n_antennas * n_subcarriers
    total = HEADER_SIZE + iq_pairs * 2
    if len(data) < total:
        raise ParseError(
            ParseErrorKind.INSUFFICIENT_DATA,
            message=f"need {total} bytes, got {len(data)}",
        )

    iq_start = HEADER_SIZE
    amps_by_ant = np.zeros((n_antennas, n_subcarriers), dtype=np.float32)
    for ant in range(n_antennas):
        for sc in range(n_subcarriers):
            off = iq_start + (ant * n_subcarriers + sc) * 2
            i_val = struct.unpack_from("b", data, off)[0]
            q_val = struct.unpack_from("b", data, off + 1)[0]
            amps_by_ant[ant, sc] = math_hypot(i_val, q_val)

    amplitude = amps_by_ant.mean(axis=0)
    frame_ts = time.time() if ts is None else ts

    frame = Esp32CsiFrame(
        ts=frame_ts,
        node_id=int(node_id),
        n_antennas=int(n_antennas),
        n_subcarriers=int(n_subcarriers),
        channel_freq_mhz=int(channel_freq_mhz),
        sequence=int(sequence),
        rssi_dbm=int(rssi_dbm),
        noise_floor_dbm=int(noise_floor_dbm),
        amplitude=amplitude,
        source_ip=source_ip,
    )
    return frame, total


def math_hypot(i: int, q: int) -> float:
    return float((i * i + q * q) ** 0.5)


def _skip_sibling_if_present(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        return offset + 1
    magic = _read_magic(data[offset:])
    if magic == ESP32_CSI_MAGIC:
        return offset
    if size := SIBLING_PACKET_SIZES.get(magic):
        return offset + size
    if ruview_sibling_packet_name(magic):
        return offset + 4
    return offset + 1


def parse_stream(data: bytes, *, source_ip: str = "") -> Tuple[List[Esp32CsiFrame], int]:
    frames: List[Esp32CsiFrame] = []
    offset = 0
    while offset < len(data):
        try:
            frame, consumed = parse_frame(data[offset:], source_ip=source_ip)
            frames.append(frame)
            offset += consumed
        except ParseError as exc:
            if exc.kind == ParseErrorKind.NON_CSI_PACKET:
                offset = _skip_sibling_if_present(data, offset)
                continue
            offset += 1
            while offset + 4 <= len(data):
                if _read_magic(data[offset:]) == ESP32_CSI_MAGIC:
                    break
                offset += 1
    return frames, offset


def build_test_frame(
    node_id: int,
    n_antennas: int,
    subcarrier_pairs: List[Tuple[int, int]],
) -> bytes:
    """Build a valid ADR-018 frame (for tests)."""
    n_sub = len(subcarrier_pairs) // n_antennas if n_antennas else len(subcarrier_pairs)
    buf = bytearray()
    buf.extend(struct.pack("<I", ESP32_CSI_MAGIC))
    buf.append(node_id & 0xFF)
    buf.append(n_antennas & 0xFF)
    buf.extend(struct.pack("<H", n_sub))
    buf.extend(struct.pack("<I", 2437))
    buf.extend(struct.pack("<I", 1))
    buf.append((-50) & 0xFF)
    buf.append((-95) & 0xFF)
    buf.extend(struct.pack("<H", 0))
    for i_val, q_val in subcarrier_pairs:
        buf.append(i_val & 0xFF)
        buf.append(q_val & 0xFF)
    return bytes(buf)
