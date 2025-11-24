# exr_utils.py
from __future__ import annotations
from pathlib import Path
import numpy as np

# Backends
_try_ox = True
try:
    import OpenEXR, Imath
except Exception:
    _try_ox = False

try:
    import imageio.v3 as iio
    _has_imgio = True
except Exception:
    _has_imgio = False

def has_openexr() -> bool:
    return _try_ox

# -------- Header / meta --------
def open_header(path: str):
    if not _try_ox:
        raise RuntimeError("OpenEXR not available.")
    f = OpenEXR.InputFile(str(path))
    return f, f.header()

def size(path: str) -> tuple[int,int]:
    if _try_ox:
        f, hdr = open_header(path)
        dw = hdr["dataWindow"]
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        return H, W
    # imageio fallback
    img = iio.imread(path)
    h, w = img.shape[:2]
    return int(h), int(w)

def list_channels(path: str) -> list[str]:
    if _try_ox:
        _, hdr = open_header(path)
        return list(hdr["channels"].keys())
    # imageio: fake channels
    img = iio.imread(path)
    if img.ndim == 2: return ["Y"]
    return ["R","G","B","A"][: img.shape[-1]]

def channel_dtype(path: str, ch: str) -> str:
    if _try_ox:
        _, hdr = open_header(path)
        pt = hdr["channels"][ch].type
        if pt == Imath.PixelType(Imath.PixelType.FLOAT): return "float32"
        if pt == Imath.PixelType(Imath.PixelType.HALF):  return "float16"
    return "float32"

# -------- Reading --------
def _choose_channel(available: list[str], prefer=("R","Y","G","B","A")) -> str:
    for p in prefer:
        if p in available: return p
    return available[0]

def read_channel(path: str, ch: str="R", prefer=("R","Y","G","B","A")) -> np.ndarray:
    p = Path(path); 
    if not p.exists(): raise FileNotFoundError(path)

    if _try_ox:
        f, hdr = open_header(str(p))
        chans = list(hdr["channels"].keys())
        use = ch if ch in chans else _choose_channel(chans, prefer)
        # read as float (promote HALF->FLOAT)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = f.channel(use, pt)
        H, W = size(str(p))
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        return arr.astype(np.float32, copy=False)

    if _has_imgio:
        img = iio.imread(str(p))
        if img.ndim == 2: return img.astype(np.float32)
        idxmap = {"R":0,"G":1,"B":2,"A":3,"Y":0}
        idx = idxmap.get(ch.upper(), 0)
        idx = min(idx, img.shape[-1]-1)
        return img[..., idx].astype(np.float32)

    raise RuntimeError("No EXR reader available (install OpenEXR/Imath or imageio).")

def read_channels(path: str, channels=("R","G","B","A")) -> dict[str,np.ndarray]:
    out = {}
    for c in channels:
        try:
            out[c] = read_channel(path, c)
        except Exception:
            pass
    return out

# -------- Writing --------
def write_single_channel(path: str, ch: str, img: np.ndarray, compression: str="ZIP") -> None:
    if not _try_ox:
        raise RuntimeError("OpenEXR required for writing.")
    img32 = np.ascontiguousarray(img.astype(np.float32))
    H, W = img32.shape
    hdr = OpenEXR.Header(W, H)
    # compression
    comp = {"ZIP": OpenEXR.ZIP_COMPRESSION, "PIZ": OpenEXR.PIZ_COMPRESSION,
            "ZIPS": OpenEXR.ZIPS_COMPRESSION, "NONE": OpenEXR.NO_COMPRESSION}.get(compression.upper(),
                                                                                   OpenEXR.ZIP_COMPRESSION)
    hdr["compression"] = comp
    hdr["channels"] = {ch: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(str(path), hdr)
    out.writePixels({ch: img32.tobytes()})
    out.close()

def write_channels(path: str, channel_map: dict[str,np.ndarray], compression: str="ZIP") -> None:
    if not _try_ox:
        raise RuntimeError("OpenEXR required for writing.")
    # assume all same shape
    ch0 = next(iter(channel_map.values()))
    H, W = ch0.shape
    hdr = OpenEXR.Header(W, H)
    hdr["compression"] = OpenEXR.ZIP_COMPRESSION
    hdr["channels"] = {k: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for k in channel_map}
    out = OpenEXR.OutputFile(str(path), hdr)
    payload = {k: np.ascontiguousarray(v.astype(np.float32)).tobytes() for k, v in channel_map.items()}
    out.writePixels(payload)
    out.close()

# -------- Depth helpers --------
def map_norm01_to_meters(alpha: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    return (min_m + np.asarray(alpha, np.float32) * (max_m - min_m)).astype(np.float32)

def meters_to_mm16(depth_m: np.ndarray) -> np.ndarray:
    mm = np.clip(np.asarray(depth_m, np.float32) * 1000.0, 0, 65535)
    return mm.astype(np.uint16)

# -------- Viz helper --------
def save_inferno_png(out_path: str, img: np.ndarray, vmin=None, vmax=None, percentiles=(2,98), invert=False):
    import matplotlib.cm as cm
    if not _has_imgio:
        raise RuntimeError("imageio required to write PNG.")
    a = np.asarray(img, np.float32)
    finite = np.isfinite(a)
    if vmin is None or vmax is None:
        lo, hi = np.percentile(a[finite], percentiles)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.nanmin(a[finite])), float(np.nanmax(a[finite]))
    norm = np.clip((a - vmin) / max(1e-12, (vmax - vmin)), 0, 1)
    if invert: norm = 1.0 - norm
    rgb = cm.get_cmap("inferno")(norm)[..., :3]
    iio.imwrite(out_path, (rgb * 255.0 + 0.5).astype(np.uint8))
    return out_path

