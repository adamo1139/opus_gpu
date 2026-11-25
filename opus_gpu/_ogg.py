"""Read one mapping-family-zero Ogg Opus stream for packet-level decoding."""

from dataclasses import dataclass
from pathlib import Path
import struct
import zlib


# Ogg uses the non-reflected CRC-32 polynomial, with an initial value of zero.
_REVERSE = bytes(int(f"{n:08b}"[::-1], 2) for n in range(256))


def _valid_crc(page, expected):
    data = (page[:22] + b"\0" * 4 + page[26:]).translate(_REVERSE)
    crc = zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF
    return (
        int.from_bytes(crc.to_bytes(4, "little").translate(_REVERSE), "big") == expected
    )


def _packet_frames(packet):
    if not packet:
        raise ValueError("empty Opus audio packet")
    toc = packet[0]
    config = toc >> 3
    if config >= 16:
        frame = 120 << (config & 3)
    elif config >= 12:
        frame = 480 << (config & 1)
    else:
        frame = (480, 960, 1920, 2880)[config & 3]
    code = toc & 3
    if code == 3 and len(packet) < 2:
        raise ValueError("missing Opus frame count")
    count = (1, 2, 2)[code] if code < 3 else packet[1] & 63
    samples = frame * count
    if not 0 < samples <= 5760:
        raise ValueError("invalid Opus packet duration")
    return samples


@dataclass
class Stream:
    packets: list[bytes]
    channels: int
    gain: int
    frames: int
    skip: int
    end: int


def read_stream(path):
    raw = Path(path).read_bytes()
    try:
        return _parse(raw)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def _parse(raw):
    pos = sequence = headers = frames = 0
    serial = origin = None
    pending = bytearray()
    packets = []
    channels = gain = skip = 0
    while pos < len(raw):
        if raw[pos : pos + 5] != b"OggS\0" or pos + 27 > len(raw):
            raise ValueError("invalid Ogg page header")
        flags, granule, page_serial, seq, crc, segments = struct.unpack_from(
            "<BqIIIB", raw, pos + 5
        )
        start = pos + 27 + segments
        lacing = raw[pos + 27 : start]
        end = start + sum(lacing)
        if end > len(raw) or not _valid_crc(raw[pos:end], crc):
            raise ValueError("truncated Ogg page or invalid checksum")
        if serial is None:
            if not flags & 2:
                raise ValueError("missing Ogg beginning of stream")
            serial = page_serial
        if page_serial != serial or seq != sequence or bool(flags & 1) != bool(pending):
            raise ValueError("multiplexed or discontinuous Ogg stream")
        sequence = (sequence + 1) & 0xFFFFFFFF
        previous = frames
        packet_start = start
        for length in lacing:
            start += length
            if length == 255:
                continue
            packet = raw[packet_start:start]
            packet_start = start
            if pending:
                pending.extend(packet)
                packet = bytes(pending)
                pending.clear()
            if headers == 0:
                if len(packet) < 19 or packet[:8] != b"OpusHead":
                    raise ValueError("missing OpusHead")
                version, channels, skip, _, gain, mapping = struct.unpack_from(
                    "<BBHIhB", packet, 8
                )
                if version >= 16 or channels not in (1, 2) or mapping != 0:
                    raise ValueError("CUDA supports mono/stereo Opus mapping family 0")
                headers += 1
            elif headers == 1:
                if not packet.startswith(b"OpusTags"):
                    raise ValueError("missing OpusTags")
                headers += 1
            else:
                frames += _packet_frames(packet)
                packets.append(packet)
        pending.extend(raw[packet_start:end])
        pos = end
        if frames > previous:
            if origin is None:
                origin = max(0, granule - frames)
            if not flags & 4 and granule != origin + frames:
                raise ValueError("invalid Ogg granule position")
        if flags & 4:
            if pos != len(raw) or pending or origin is None:
                raise ValueError("incomplete or chained Ogg Opus stream")
            stop = granule - origin
            if not max(skip, previous) <= stop <= frames:
                raise ValueError("invalid final Ogg granule position")
            return Stream(packets, channels, gain, frames, skip, stop)
    raise ValueError("missing Ogg end of stream")
