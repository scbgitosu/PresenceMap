"""Atheros CSI Tool frame reader.

Parses the binary frame format produced by ``recvCSI`` (Atheros-CSI-Tool /
Yaxiong Xie fork) and exposes both a real ``AtherosCSIReader`` (subprocess
or UDP source) and a ``FixtureCSISource`` for testing on platforms without
the AR9271 hardware.

Frame format (little-endian unless noted):

    uint64  tstamp           hardware timestamp (ns since boot)
    uint16  csi_len          total CSI bit-packed length, in bytes
    uint16  tx_channel       channel/frequency code
    uint8   err_info
    uint8   noise_floor      noise in dBm (signed-as-uint8)
    uint8   rate             MCS rate index
    uint8   bandwidth        0 = HT20 (56 tones), 1 = HT40 (114 tones)
    uint8   num_tones
    uint8   nr               # rx antennas
    uint8   nc               # tx antennas
    uint8   rssi             combined RSSI
    uint8   rssi_1           per-chain RSSI
    uint8   rssi_2
    uint8   rssi_3
    uint8   payload_len
    bytes[csi_len]           CSI payload (bit-packed, see unpack_csi_payload)

Reference: https://github.com/xieyaxiongfly/Atheros-CSI-Tool
"""
from __future__ import annotations

import math
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Protocol

import numpy as np

# Header struct: <Q H H 12B = 8+2+2+12 = 24 bytes
_HEADER_STRUCT = struct.Struct("<QHHBBBBBBBBBBBB")
HEADER_SIZE = _HEADER_STRUCT.size
assert HEADER_SIZE == 24, "header struct mismatch"

BITS_PER_SYMBOL = 10  # signed 10-bit real/imag values


@dataclass
class CSIFrame:
    ts: float                  # seconds since epoch when this frame arrived
    csi: np.ndarray            # complex64, shape (nr, nc, num_tones)
    nr: int
    nc: int
    num_tones: int
    bandwidth: int             # 0 = HT20, 1 = HT40
    rssi: int                  # combined
    rssi_1: int
    rssi_2: int
    rssi_3: int
    noise_dbm: int             # NB: serialized as signed-as-uint8
    rate: int
    tstamp_ns: int             # raw hardware timestamp


def _signbit_convert(value: int, nbits: int) -> int:
    """Reinterpret an ``nbits``-wide unsigned value as a signed integer."""
    if value & (1 << (nbits - 1)):
        return value - (1 << nbits)
    return value


def _u8_signed(value: int) -> int:
    """The Atheros CSI tool stores noise as a signed value packed in a uint8."""
    return value - 256 if value >= 128 else value


def unpack_csi_payload(payload: bytes, *, nr: int, nc: int, num_tones: int) -> np.ndarray:
    """Unpack the bit-packed payload into a complex64 array of shape (nr, nc, num_tones).

    The packing is LSB-first across the byte stream: for each subcarrier index
    in [0..num_tones), and each (rx, tx) antenna pair, two 10-bit signed
    integers are read (real then imaginary). Matches Xie's ``read_csi`` MATLAB
    reference implementation.
    """
    needed_bits = num_tones * nr * nc * BITS_PER_SYMBOL * 2
    if len(payload) * 8 < needed_bits:
        raise ValueError(
            f"payload too short: need {needed_bits} bits, got {len(payload) * 8}"
        )

    out = np.zeros((nr, nc, num_tones), dtype=np.complex64)
    bit_index = 0

    def read_bits(n: int) -> int:
        nonlocal bit_index
        v = 0
        for i in range(n):
            byte_idx = bit_index >> 3
            in_byte = bit_index & 7
            v |= ((payload[byte_idx] >> in_byte) & 1) << i
            bit_index += 1
        return v

    for k in range(num_tones):
        for c in range(nc):
            for r in range(nr):
                real_u = read_bits(BITS_PER_SYMBOL)
                imag_u = read_bits(BITS_PER_SYMBOL)
                real = _signbit_convert(real_u, BITS_PER_SYMBOL)
                imag = _signbit_convert(imag_u, BITS_PER_SYMBOL)
                out[r, c, k] = complex(real, imag)
    return out


def pack_csi_payload(csi: np.ndarray) -> bytes:
    """Inverse of ``unpack_csi_payload``; used by tests/fixtures."""
    nr, nc, num_tones = csi.shape
    bits = bytearray((num_tones * nr * nc * BITS_PER_SYMBOL * 2 + 7) // 8)
    bit_index = 0

    def write_bits(value: int, n: int) -> None:
        nonlocal bit_index
        # Mask to n bits in two's-complement form.
        if value < 0:
            value = value + (1 << n)
        value &= (1 << n) - 1
        for i in range(n):
            bit = (value >> i) & 1
            byte_idx = bit_index >> 3
            in_byte = bit_index & 7
            if bit:
                bits[byte_idx] |= 1 << in_byte
            bit_index += 1

    for k in range(num_tones):
        for c in range(nc):
            for r in range(nr):
                real = int(round(csi[r, c, k].real))
                imag = int(round(csi[r, c, k].imag))
                # clip to signed 10-bit range
                real = max(-(1 << (BITS_PER_SYMBOL - 1)), min((1 << (BITS_PER_SYMBOL - 1)) - 1, real))
                imag = max(-(1 << (BITS_PER_SYMBOL - 1)), min((1 << (BITS_PER_SYMBOL - 1)) - 1, imag))
                write_bits(real, BITS_PER_SYMBOL)
                write_bits(imag, BITS_PER_SYMBOL)
    return bytes(bits)


def parse_csi_frame(buf: bytes, offset: int = 0) -> tuple[CSIFrame, int]:
    """Parse one frame starting at ``offset``. Returns (frame, next_offset)."""
    if len(buf) - offset < HEADER_SIZE:
        raise ValueError("buffer truncated before header")
    (
        tstamp,
        csi_len,
        tx_channel,
        err_info,
        noise_floor,
        rate,
        bandwidth,
        num_tones,
        nr,
        nc,
        rssi,
        rssi_1,
        rssi_2,
        rssi_3,
        payload_len,
    ) = _HEADER_STRUCT.unpack_from(buf, offset)
    body_start = offset + HEADER_SIZE
    body_end = body_start + csi_len
    if body_end > len(buf):
        raise ValueError(
            f"buffer truncated: header says csi_len={csi_len} but only {len(buf) - body_start} bytes left"
        )
    payload = buf[body_start:body_end]
    csi = (
        unpack_csi_payload(payload, nr=nr, nc=nc, num_tones=num_tones)
        if (nr and nc and num_tones)
        else np.zeros((max(nr, 1), max(nc, 1), max(num_tones, 1)), dtype=np.complex64)
    )
    frame = CSIFrame(
        ts=time.time(),
        csi=csi,
        nr=nr,
        nc=nc,
        num_tones=num_tones,
        bandwidth=bandwidth,
        rssi=rssi,
        rssi_1=rssi_1,
        rssi_2=rssi_2,
        rssi_3=rssi_3,
        noise_dbm=_u8_signed(noise_floor),
        rate=rate,
        tstamp_ns=tstamp,
    )
    return frame, body_end


def build_csi_frame_bytes(
    *,
    tstamp_ns: int,
    nr: int,
    nc: int,
    num_tones: int,
    csi: np.ndarray,
    bandwidth: int = 0,
    rssi: int = 200,
    rssi_1: int = 200,
    rssi_2: int = 0,
    rssi_3: int = 0,
    noise_dbm: int = -95,
    rate: int = 7,
    tx_channel: int = 2437,
    err_info: int = 0,
) -> bytes:
    """Construct a CSI binary frame for tests / fixtures."""
    payload = pack_csi_payload(csi.astype(np.complex64))
    noise_u8 = noise_dbm + 256 if noise_dbm < 0 else noise_dbm
    header = _HEADER_STRUCT.pack(
        tstamp_ns,
        len(payload),
        tx_channel,
        err_info,
        noise_u8 & 0xFF,
        rate,
        bandwidth,
        num_tones,
        nr,
        nc,
        rssi & 0xFF,
        rssi_1 & 0xFF,
        rssi_2 & 0xFF,
        rssi_3 & 0xFF,
        len(payload) & 0xFF,
    )
    return header + payload


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


class CSISource(Protocol):
    def frames(self) -> Iterator[CSIFrame]: ...
    def close(self) -> None: ...


class FixtureCSISource:
    """Yields pre-built ``CSIFrame``s at a fixed rate. Useful for tests and the
    Mac-side dev environment where there's no AR9271.
    """

    def __init__(self, frames: List[CSIFrame], rate_hz: float = 30.0) -> None:
        self._frames = list(frames)
        self._rate_hz = rate_hz
        self._stop = threading.Event()

    def frames(self) -> Iterator[CSIFrame]:
        period = 1.0 / max(self._rate_hz, 0.1)
        i = 0
        while not self._stop.is_set():
            f = self._frames[i % len(self._frames)]
            # Refresh timestamp so feature aggregation sees a moving window.
            yield CSIFrame(
                ts=time.time(),
                csi=f.csi,
                nr=f.nr,
                nc=f.nc,
                num_tones=f.num_tones,
                bandwidth=f.bandwidth,
                rssi=f.rssi,
                rssi_1=f.rssi_1,
                rssi_2=f.rssi_2,
                rssi_3=f.rssi_3,
                noise_dbm=f.noise_dbm,
                rate=f.rate,
                tstamp_ns=f.tstamp_ns + i,
            )
            i += 1
            self._stop.wait(period)

    def close(self) -> None:
        self._stop.set()


class AtherosCSIReader:
    """Reads frames from the Atheros CSI Tool via UDP or a subprocess pipe.

    The recommended deployment runs a stdout-streaming fork of ``recvCSI`` and
    we read its stdout. For UDP-based forks set ``udp_port`` to the recvCSI
    publish port (default 9300).
    """

    def __init__(
        self,
        *,
        binary: Optional[str] = None,
        binary_args: Optional[list] = None,
        udp_host: Optional[str] = None,
        udp_port: Optional[int] = None,
        on_health: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if not binary and not udp_port:
            raise ValueError("must provide either binary or udp_port")
        self._binary = binary
        self._binary_args = binary_args or []
        self._udp_host = udp_host or "127.0.0.1"
        self._udp_port = udp_port
        self._proc: Optional[subprocess.Popen] = None
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._stop = threading.Event()
        self._on_health = on_health or (lambda code, detail: None)

    def open(self) -> None:
        if self._binary is not None:
            self._proc = subprocess.Popen(
                [self._binary, *self._binary_args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind((self._udp_host, int(self._udp_port)))  # type: ignore[arg-type]
            self._sock.settimeout(1.0)

    def frames(self) -> Iterator[CSIFrame]:
        if self._proc is None and self._sock is None:
            self.open()
        while not self._stop.is_set():
            chunk = self._read_chunk()
            if not chunk:
                continue
            self._buf += chunk
            # Try to parse as many complete frames as we have buffered.
            offset = 0
            while True:
                if len(self._buf) - offset < HEADER_SIZE:
                    break
                # Peek csi_len without unpacking the whole header twice.
                csi_len = int.from_bytes(self._buf[offset + 8 : offset + 10], "little")
                if len(self._buf) - offset < HEADER_SIZE + csi_len:
                    break
                try:
                    frame, next_off = parse_csi_frame(self._buf, offset)
                except ValueError as exc:
                    self._on_health("csi_parse_error", str(exc))
                    # Skip the corrupt header to make forward progress.
                    next_off = offset + HEADER_SIZE
                else:
                    yield frame
                offset = next_off
            self._buf = self._buf[offset:]

    def _read_chunk(self) -> bytes:
        if self._proc is not None:
            try:
                return self._proc.stdout.read1(65536)  # type: ignore[union-attr]
            except Exception as exc:
                self._on_health("csi_pipe_error", str(exc))
                return b""
        try:
            data, _ = self._sock.recvfrom(65536)  # type: ignore[union-attr]
            return data
        except socket.timeout:
            return b""

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
