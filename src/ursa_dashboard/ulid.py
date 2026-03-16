from __future__ import annotations

import secrets
import time

# Crockford's Base32 (no I, L, O, U)
_CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _encode_base32(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_CROCKFORD32[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def new_ulid() -> str:
    """Generate a ULID string (26 chars) per spec:

    - 48 bits: timestamp in milliseconds
    - 80 bits: randomness

    Returns an uppercase Crockford Base32 ULID.
    """
    ts_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(secrets.token_bytes(10), "big")  # 80 bits
    # Combine to 128-bit integer
    value = (ts_ms << 80) | rand
    # ULID encodes 128 bits into 26 base32 chars (130 bits), with leading zero padding.
    return _encode_base32(value, 26)
