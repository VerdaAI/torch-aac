"""JIT-compiled C BitWriter loaded via ctypes.

Compiles ``_bitwriter.c`` on first import using the system C compiler, caches
the shared library next to the source file, and exposes ``bitwriter_pack()``
for fast bit-packing of pre-computed (code, length) arrays.

Falls back gracefully: if the compiler isn't available or compilation fails,
``bitwriter_pack`` is set to ``None`` and callers should use the pure-Python
``BitWriter`` instead.
"""

from __future__ import annotations

import ctypes
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_LIB: ctypes.CDLL | None = None

# Paths
_C_SRC = Path(__file__).parent / "_bitwriter.c"
_SUFFIX = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(platform.system(), ".so")
_LIB_PATH = _C_SRC.with_suffix(_SUFFIX)


def _compile() -> ctypes.CDLL | None:
    """Compile the C source to a shared library. Returns None on failure."""
    if _LIB_PATH.exists() and _LIB_PATH.stat().st_mtime >= _C_SRC.stat().st_mtime:
        try:
            return ctypes.CDLL(str(_LIB_PATH))
        except OSError:
            pass  # stale/corrupt .so — recompile

    cc = "cc"  # POSIX default; gcc/clang on Linux/macOS
    flags = ["-O2", "-shared", "-fPIC"]

    if platform.system() == "Darwin":
        flags.append("-undefined")
        flags.append("dynamic_lookup")

    cmd = [cc, *flags, "-o", str(_LIB_PATH), str(_C_SRC)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            print(
                f"[torch_aac] C BitWriter compile failed: {result.stderr.decode()[:200]}",
                file=sys.stderr,
            )
            return None
        return ctypes.CDLL(str(_LIB_PATH))
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[torch_aac] C BitWriter compile skipped ({e})", file=sys.stderr)
        return None


def _load() -> ctypes.CDLL | None:
    global _LIB
    if _LIB is None:
        _LIB = _compile()
        if _LIB is not None:
            # Set up argtypes / restype for bitwriter_pack
            _LIB.bitwriter_pack.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),  # codes
                ctypes.POINTER(ctypes.c_uint8),  # lengths
                ctypes.c_int,  # n
                ctypes.POINTER(ctypes.c_uint8),  # output
                ctypes.c_int,  # capacity
            ]
            _LIB.bitwriter_pack.restype = ctypes.c_int

            _LIB.bitwriter_pack_ics.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
            ]
            _LIB.bitwriter_pack_ics.restype = ctypes.c_int
    return _LIB


def bitwriter_pack(
    codes: NDArray[np.uint32],
    lengths: NDArray[np.uint8],
    output: NDArray[np.uint8],
) -> int:
    """Pack (code, length) pairs into a byte buffer using the C BitWriter.

    Args:
        codes: uint32 array of codewords, contiguous.
        lengths: uint8 array of bit lengths, contiguous, same length as codes.
        output: Pre-allocated uint8 output buffer (large enough).

    Returns:
        Total number of bits written.

    Raises:
        RuntimeError: If the C library is not available.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError("C BitWriter not available — compilation failed")

    codes = np.ascontiguousarray(codes, dtype=np.uint32)
    lengths = np.ascontiguousarray(lengths, dtype=np.uint8)
    output = np.ascontiguousarray(output, dtype=np.uint8)
    n = len(codes)

    total_bits = lib.bitwriter_pack(
        codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        n,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        len(output),
    )
    return int(total_bits)


def is_available() -> bool:
    """Check whether the C BitWriter compiled successfully."""
    return _load() is not None
