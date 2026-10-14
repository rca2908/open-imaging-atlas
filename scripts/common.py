"""Shared helpers for packing atlas studies into web-servable files.

Conventions for every study written here:
  * Volumes are stored x-fastest with index axes (i, j, k) meaning
    i -> towards patient LEFT, j -> towards patient POSTERIOR, k -> towards SUPERIOR
    (DICOM LPS). numpy arrays are therefore shaped (k, j, i).
  * Binary payloads are stored as 8-bit grayscale PNGs, 4096 px wide, one byte
    per pixel, because artifacts only serve web file types. meta.json carries the
    byte length and Adler-32 so the page can verify the decode.
"""
import json, os, struct, zlib
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import measure

PNG_W = 4096
MAX_PNG_BYTES = 12_000_000  # split large payloads into several PNGs


def write_payload(outdir, name, raw: bytes):
    """Write raw bytes as one or more grayscale PNGs; return the file entry."""
    parts, chunk = [], PNG_W * 2048  # 8 MiB of raw bytes per part
    for p, start in enumerate(range(0, len(raw), chunk)):
        piece = raw[start:start + chunk]
        h = -(-len(piece) // PNG_W)
        arr = np.zeros(PNG_W * h, np.uint8)
        arr[:len(piece)] = np.frombuffer(piece, np.uint8)
        fn = f"{name}.{p}.png"
        Image.fromarray(arr.reshape(h, PNG_W), "L").save(os.path.join(outdir, fn), optimize=True, compress_level=9)
        parts.append({"file": fn, "bytes": len(piece)})
    size = sum(os.path.getsize(os.path.join(outdir, x["file"])) for x in parts)
    assert all(os.path.getsize(os.path.join(outdir, x["file"])) < 15_000_000 for x in parts)
    return {"parts": parts, "bytes": len(raw), "adler32": zlib.adler32(raw), "png_bytes": size}


# ---- CT companding: 8 bits, fine steps where soft tissue lives -------------
CT_KNOTS = [(-1024, 0), (-200, 40), (300, 215), (2000, 255)]


def ct_to_u8(hu):
    xs, ys = zip(*CT_KNOTS)
    return np.rint(np.interp(np.clip(hu, -1024, 2000), xs, ys)).astype(np.uint8)


def ct_decode_table():
    """HU value for each stored byte (used by the page to apply windows)."""
    xs, ys = zip(*CT_KNOTS)
    return [round(float(np.interp(v, ys, xs)), 1) for v in range(256)]


def meshes_payload(labels, spacing, ids, big_voxels=60000, sigma=0.8):
    """Marching-cubes surfaces for each label id. Vertices in mm (u16 * 1/16 mm)."""
    buf = bytearray(struct.pack("<I", 0))
    count, tris = 0, 0
    sp = np.array(spacing[::-1], float)  # arrays are (k, j, i)
    for sid in ids:
        m = labels == sid
        n = int(m.sum())
        if n < 30:
            continue
        sl = ndimage.find_objects(m.astype(np.uint8))[0]
        sl = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in sl)
        sub = ndimage.gaussian_filter(m[sl].astype(np.float32), sigma)
        step = 3 if n > big_voxels else (2 if n > 3000 else 1)
        try:
            v, f, _, _ = measure.marching_cubes(sub, 0.5, step_size=step, spacing=tuple(sp))
        except (ValueError, RuntimeError):
            continue
        off = np.array([s.start for s in sl]) * sp
        v = (v + off)[:, ::-1]  # -> (x, y, z) mm
        if len(v) >= 65536:
            idx_fmt, isz = "<u4", 4
        else:
            idx_fmt, isz = "<u2", 2
        buf += struct.pack("<BBII", sid, isz, len(v), len(f))
        buf += np.clip(np.rint(v * 16), 0, 65535).astype("<u2").tobytes()
        buf += f.astype(idx_fmt).tobytes()
        if isz == 2 and (len(f) * 3 * 2) % 4:
            buf += b"\0\0"  # keep 4-byte alignment
        count += 1
        tris += len(f)
    struct.pack_into("<I", buf, 0, count)
    return bytes(buf), count, tris


def structure_stats(labels, spacing, ids):
    """Voxel count, volume in cm3 and a 'go to' voxel inside each structure."""
    out = {}
    vox_ml = float(np.prod(spacing)) / 1000.0
    objs = ndimage.find_objects(labels)
    for sid in ids:
        if sid > len(objs) or objs[sid - 1] is None:
            continue
        sl = objs[sid - 1]
        m = labels[sl] == sid
        pts = np.argwhere(m)
        c = pts.mean(0)
        best = pts[np.argmin(((pts - c) ** 2).sum(1))] + [s.start for s in sl]
        out[sid] = {"voxels": int(m.sum()), "ml": round(int(m.sum()) * vox_ml, 2),
                    "center": [int(best[2]), int(best[1]), int(best[0])]}
    return out


def write_meta(outdir, meta):
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, separators=(",", ":"))


def seg_frames_to_grid(sd, grid, value=1, out=None):
    """Paint a BINARY DICOM SEG onto an axis-aligned SimpleITK grid, honouring the
    SEG's own orientation (rows may run in -y), position and pixel spacing."""
    import numpy as _np
    size = grid.GetSize()
    if out is None:
        out = _np.zeros(size[::-1], _np.uint8)
    frames = sd.pixel_array
    if frames.ndim == 2:
        frames = frames[None]
    sh = sd.SharedFunctionalGroupsSequence[0]
    iop = [float(v) for v in sh.PlaneOrientationSequence[0].ImageOrientationPatient]
    ps = [float(v) for v in sh.PixelMeasuresSequence[0].PixelSpacing]  # row spacing, col spacing
    rowdir, coldir = _np.array(iop[:3]), _np.array(iop[3:])  # along a row (col index), down a column (row index)
    for f, fg in zip(frames, sd.PerFrameFunctionalGroupsSequence):
        ipp = _np.array([float(v) for v in fg.PlanePositionSequence[0].ImagePositionPatient])
        rr, cc = _np.nonzero(f)
        if not len(rr):
            continue
        pts = ipp[None] + cc[:, None] * ps[1] * rowdir[None] + rr[:, None] * ps[0] * coldir[None]
        org, spc = _np.array(grid.GetOrigin()), _np.array(grid.GetSpacing())
        d = _np.array(grid.GetDirection()).reshape(3, 3)
        ijk = _np.rint(((pts - org) @ d) / spc).astype(int)
        ok = (ijk >= 0).all(1) & (ijk < _np.array(size)).all(1)
        ijk = ijk[ok]
        out[ijk[:, 2], ijk[:, 1], ijk[:, 0]] = value
    return out
