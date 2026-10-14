"""Build the brain MRI, prostate MRI and prostate transrectal US studies.

usage: build_mr_us.py <study> <outdir>   study in {brain, prostate, trus}
"""
import glob, gzip, json, os, struct, sys, zlib
import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image
from scipy import ndimage
from skimage import measure

sys.path.insert(0, os.path.dirname(__file__))
from common import write_payload, meshes_payload, structure_stats, write_meta, seg_frames_to_grid

STUDY, OUT = sys.argv[1], sys.argv[2]
ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUT, exist_ok=True)


def read_png_payload(meta, name, d):
    px = np.asarray(Image.open(f"{d}/{name}.png")).reshape(-1)
    return px[: meta["files"][name]["bytes"]]


def u8(v, lo_pct=0.5, hi_pct=99.7, mask=None):
    ref = v[mask] if mask is not None else v[v > 0]
    lo, hi = np.percentile(ref, lo_pct), np.percentile(ref, hi_pct)
    return np.clip((v.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


if STUDY == "brain":
    # Re-use the arrays built for the first version (RAS index order) and flip to LPS.
    old = f"{ROOT}/build/brain_v1"  # output of build_brain_v1.py + pack_png_v1.py
    meta0 = json.load(open(f"{old}/meta.json"))
    nx, ny, nz = meta0["dims"]
    get = lambda n: read_png_payload(meta0, n, old).reshape(nz, ny, nx)[:, ::-1, ::-1].copy()
    t1, t2, lab = get("t1"), get("t2"), get("labels")
    sp = (1.0, 1.0, 1.0)
    structs = meta0["structures"]
    for s in structs:
        s.pop("src", None)
        c = s["center"]; s["center"] = [nx - 1 - c[0], ny - 1 - c[1], c[2]]
        s["ml"] = round(s["voxels"] / 1000, 2)
    files = {"t1": write_payload(OUT, "t1", t1.tobytes()), "t2": write_payload(OUT, "t2", t2.tobytes()),
             "labels": write_payload(OUT, "labels", lab.tobytes())}
    ids = [s["id"] for s in structs if s["group"] != "Cerebral cortex"
           and s["name"] not in ("Cerebral white matter", "Extracerebral CSF")]
    raw, n, ntri = meshes_payload(lab, sp, ids)
    # cortical shell coloured by nearest parcel (id 0 + per-vertex labels)
    ctx_ids = [s["id"] for s in structs if s["group"] == "Cerebral cortex"]
    cer_ids = ctx_ids + [s["id"] for s in structs if s["group"] in ("White matter", "Deep grey matter")
                         or s["name"] in ("Lateral ventricle", "Inferior horn of lateral ventricle", "Third ventricle", "Choroid plexus")]
    cer = ndimage.binary_fill_holes(ndimage.binary_closing(np.isin(lab, cer_ids), iterations=2))
    v, f, _, _ = measure.marching_cubes(ndimage.gaussian_filter(cer.astype(np.float32), 1.2), 0.5, step_size=2)
    ctx_only = np.where(np.isin(lab, ctx_ids), lab, 0)
    _, idx = ndimage.distance_transform_edt(ctx_only == 0, return_indices=True)
    vi = np.clip(np.rint(v).astype(int), 0, np.array(lab.shape) - 1)
    vl = ctx_only[idx[0][vi[:, 0], vi[:, 1], vi[:, 2]], idx[1][vi[:, 0], vi[:, 1], vi[:, 2]], idx[2][vi[:, 0], vi[:, 1], vi[:, 2]]]
    v = v[:, ::-1]
    buf = bytearray(raw)
    buf += struct.pack("<BBII", 0, 4, len(v), len(f))
    buf += np.rint(v * 16).astype("<u2").tobytes()
    buf += f.astype("<u4").tobytes()
    buf += vl.astype(np.uint8).tobytes()
    struct.pack_into("<I", buf, 0, n + 1)
    files["meshes"] = write_payload(OUT, "meshes", bytes(buf))
    meta = {"kind": "mr", "dims": [nx, ny, nz], "spacing": list(sp), "origin": [-o for o in meta0["origin"][:2]] + [meta0["origin"][2]],
            "channels": ["t1", "t2"], "files": files, "structures": structs,
            "groups": ["Cerebral cortex", "Deep grey matter", "White matter", "Ventricles and CSF", "Brainstem and cerebellum"],
            "coords": "mni"}
    # MNI x of voxel i (LPS index): x = -(i*1 + origin_lps) ... keep simple: store RAS origin for display
    meta["mni_origin"] = meta0["origin"]
    meta["mni_dims"] = [nx, ny, nz]

elif STUDY == "prostate":
    mdir = glob.glob(f"{ROOT}/dicom/prostate/MR_*")[0]
    r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(mdir))
    mr = r.Execute()
    arr = sitk.GetArrayFromImage(mr).astype(np.float32)  # (k, j, i)
    lab = np.zeros(arr.shape, np.uint8)
    for sid, pattern in ((1, "Segmentation of prostate"), (2, "Segmentation of prostate lesion 1")):
        for fpath in glob.glob(f"{ROOT}/dicom/prostate/SEG_*/*"):
            sd = pydicom.dcmread(fpath)
            if sd.SeriesDescription != pattern:
                continue
            m = seg_frames_to_grid(sd, mr)
            lab[m > 0] = sid
    sp = mr.GetSpacing()
    structs = [
        {"id": 1, "name": "Prostate gland", "group": "Pelvis", "hemi": "M", "color": [230, 158, 140],
         "note": "Outlined by a radiologist for fusion biopsy. Peripheral zone posteriorly, transition zone around the urethra."},
        {"id": 2, "name": "Biopsy target lesion", "group": "Lesions", "hemi": "M", "color": [255, 80, 60],
         "note": "Region outlined on MRI as suspicious and targeted at MRI/ultrasound fusion biopsy."},
    ]
    st = structure_stats(lab, sp, [1, 2])
    for s in structs:
        s.update(st[s["id"]])
    files = {"t2": write_payload(OUT, "t2", u8(arr).tobytes()), "labels": write_payload(OUT, "labels", lab.tobytes())}
    raw, n, ntri = meshes_payload(lab, sp, [1, 2], sigma=0.6)
    files["meshes"] = write_payload(OUT, "meshes", raw)
    meta = {"kind": "mr", "dims": list(arr.shape[::-1]), "spacing": list(sp), "origin": list(mr.GetOrigin()),
            "channels": ["t2"], "files": files, "structures": structs, "groups": ["Pelvis", "Lesions"]}

elif STUDY == "trus":
    f = sorted(glob.glob(f"{ROOT}/dicom/prostate/US_*/*"))[0]
    a = pydicom.dcmread(f).pixel_array  # (depth, a, b), probe at depth 0
    a = ndimage.zoom(a.astype(np.float32), 0.5, order=1)  # 0.356 -> 0.712 mm
    a = a[::-1]  # put the probe at the top of the through-probe planes
    vol = np.clip(a, 0, 255).astype(np.uint8)
    sp = (0.712, 0.712, 0.712)
    files = {"us": write_payload(OUT, "us", vol.tobytes())}
    meta = {"kind": "us", "dims": list(vol.shape[::-1]), "spacing": list(sp), "origin": [0, 0, 0],
            "channels": ["us"], "files": files, "structures": [], "groups": []}

write_meta(OUT, meta)
print(STUDY, meta["dims"], {k: round(v["png_bytes"] / 1e6, 2) for k, v in files.items()})
