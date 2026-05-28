"""
image_io.py -- format-agnostic image loading.

Returns float64 NumPy arrays of shape (H, W, 3) with values in [0, 1].
Loaders are tried in order:
  1. PIL / Pillow  (pip install Pillow)
  2. imageio       (pip install imageio)
  3. Built-in P5/P6 PPM/PGM reader (no extra deps, binary formats only)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Union


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image and return it as a float64 array of shape (H, W, 3) in [0, 1].

    RGB channel order is always preserved.  Greyscale inputs are broadcast to
    three identical channels.  Alpha channels are dropped.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    ndarray, shape (H, W, 3), dtype float64, values in [0, 1]

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If no available loader can handle the file format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    # --- attempt 1: PIL / Pillow ---
    try:
        from PIL import Image  # type: ignore
        img = Image.open(path).convert("RGB")
        return np.asarray(img, dtype=np.float64) / 255.0
    except ImportError:
        pass
    except Exception:
        pass  # try next loader

    # --- attempt 2: imageio ---
    try:
        import imageio  # type: ignore
        arr = imageio.v3.imread(str(path))
        return _normalise_array(arr)
    except ImportError:
        pass
    except AttributeError:
        try:
            import imageio  # type: ignore
            arr = imageio.imread(str(path))
            return _normalise_array(arr)
        except Exception:
            pass
    except Exception:
        pass

    # --- attempt 3: built-in PPM / PGM reader ---
    suffix = path.suffix.lower()
    if suffix in (".ppm", ".pgm", ".pbm", ".pnm"):
        return _load_netpbm(path)

    raise RuntimeError(
        f"Cannot load '{path}': no compatible loader found. "
        "Install Pillow (pip install Pillow) or imageio (pip install imageio), "
        "or convert the file to PPM/PGM format."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_array(arr: np.ndarray) -> np.ndarray:
    """Convert an ndarray from any integer depth or float to float64 [0,1], (H,W,3)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Unexpected channel count {arr.shape[2]}")

    arr = arr.astype(np.float64)
    if arr.max() > 1.0:
        # Determine bit depth from dtype
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            arr = arr / info.max
        else:
            arr = arr / arr.max()
    return arr


def _load_netpbm(path: Path) -> np.ndarray:
    """
    Pure-Python reader for binary PBM/PGM/PPM (P4/P5/P6) files.

    Returns float64 (H, W, 3) in [0, 1].
    """
    with open(path, "rb") as f:
        raw = f.read()

    # Parse header: 3 tokens for P4/P5 (magic, W, H) + maxval for P5/P6,
    # or 4 tokens total for P6 (magic, W, H, maxval).
    idx = 0
    tokens: list[str] = []

    while len(tokens) < 4:
        # Skip whitespace and comment lines
        while idx < len(raw):
            ch = raw[idx : idx + 1]
            if ch == b"#":
                while idx < len(raw) and raw[idx : idx + 1] != b"\n":
                    idx += 1
            elif ch in (b" ", b"\t", b"\n", b"\r"):
                idx += 1
            else:
                break

        if idx >= len(raw):
            break

        # Read one token
        start = idx
        while idx < len(raw) and raw[idx : idx + 1] not in (b" ", b"\t", b"\n", b"\r"):
            idx += 1
        tokens.append(raw[start:idx].decode("ascii"))

        # P4 (bitmap) and P5 (greyscale) have only 3 header tokens
        if len(tokens) == 1 and tokens[0] == "P4":
            # PBM has no maxval; treat remaining 2 tokens as W, H then stop
            pass
        if len(tokens) == 3 and tokens[0] == "P4":
            break  # no maxval for bitmap

    magic = tokens[0]
    if magic not in ("P4", "P5", "P6"):
        raise ValueError(
            f"Only binary NetPBM formats (P4/P5/P6) are supported; got '{magic}'"
        )

    width, height = int(tokens[1]), int(tokens[2])
    maxval = int(tokens[3]) if magic != "P4" else 1

    # One whitespace byte separates header from pixel data
    idx += 1

    if magic == "P6":
        dtype = np.uint8 if maxval <= 255 else np.uint16
        if maxval > 255:
            # Big-endian 16-bit PPM
            data = np.frombuffer(raw[idx:], dtype=">u2").astype(np.float64)
        else:
            data = np.frombuffer(raw[idx:], dtype=np.uint8).astype(np.float64)
        arr = data.reshape(height, width, 3) / maxval
        return arr

    elif magic == "P5":
        dtype = np.uint8 if maxval <= 255 else np.uint16
        if maxval > 255:
            data = np.frombuffer(raw[idx:], dtype=">u2").astype(np.float64)
        else:
            data = np.frombuffer(raw[idx:], dtype=np.uint8).astype(np.float64)
        grey = data.reshape(height, width) / maxval
        return np.stack([grey, grey, grey], axis=-1)

    else:  # P4 -- 1-bit bitmap
        n_bytes = (width + 7) // 8
        rows = []
        for r in range(height):
            row_bytes = raw[idx : idx + n_bytes]
            idx += n_bytes
            bits = np.unpackbits(np.frombuffer(row_bytes, dtype=np.uint8))[:width]
            rows.append(bits)
        grey = np.array(rows, dtype=np.float64)
        return np.stack([grey, grey, grey], axis=-1)
