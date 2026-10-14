"""Build CT studies (whole-body anatomy and PET-CT pathology cases).

usage: build_ct.py <case> <outdir>   case in {neg, lym, lung}
Inputs (from ../nii and ../dicom): <case>_ct.nii.gz, <case>_seg.nii.gz (TotalSegmentator
--ml --fast), <case>_pt.nii.gz and the lesion DICOM SEG for the pathology cases.
"""
import glob, json, os, sys
from datetime import datetime
import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage
from totalsegmentator.map_to_binary import class_map

sys.path.insert(0, os.path.dirname(__file__))
from common import (write_payload, ct_to_u8, ct_decode_table, meshes_payload,
                    structure_stats, write_meta, seg_frames_to_grid)
from ts_names import describe, GROUP_ORDER

CASE, OUT = sys.argv[1], sys.argv[2]
ROOT = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUT, exist_ok=True)
SPACING = (2.0, 2.0, 2.5)  # mm, (i, j, k)

ref = sitk.ReadImage(f"{ROOT}/nii/{CASE}_ct2.nii.gz")   # cropped, 2 x 2 x 2.5 mm (prep_ct.py)
seg = sitk.ReadImage(f"{ROOT}/nii/{CASE}_seg.nii.gz")    # TotalSegmentator on that grid

def resample(img, interp, default=0, pix=None):
    f = sitk.ResampleImageFilter()
    f.SetReferenceImage(ref); f.SetInterpolator(interp); f.SetDefaultPixelValue(default)
    if pix is not None:
        f.SetOutputPixelType(pix)
    return sitk.GetArrayFromImage(f.Execute(img))

hu = sitk.GetArrayFromImage(ref).astype(np.float32)
ts = resample(seg, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8).astype(np.uint8)
shape = hu.shape  # (k, j, i)
print("grid", shape[::-1], "voxels", hu.size)

# ---- structures ------------------------------------------------------------
names = class_map["total"]
present = [int(v) for v in np.unique(ts) if v]
structs = []
for sid in present:
    d = describe(names[sid])
    structs.append({"id": sid, "raw": names[sid], **{k: v for k, v in d.items() if k != "order"},
                    "order": str(d.get("order", ""))})
stats = structure_stats(ts, SPACING, present)
for s in structs:
    s.update(stats.get(s["id"], {}))
structs = [s for s in structs if s.get("voxels", 0) >= 20]

# ---- regions for quick navigation (z index ranges) -------------------------
def zrange(raws):
    ids = [k for k, v in names.items() if v in raws or any(v.startswith(r) for r in raws if r.endswith("_"))]
    m = np.isin(ts, ids)
    zs = np.where(m.any((1, 2)))[0]
    return (int(zs.min()), int(zs.max())) if len(zs) else None

REGIONS = [
    ("Head", ["brain", "skull"]),
    ("Neck", ["vertebrae_C", "thyroid_gland", "trachea"]),
    ("Thorax", ["lung_", "heart"]),
    ("Abdomen", ["liver", "spleen", "kidney_", "pancreas", "stomach"]),
    ("Pelvis", ["hip_", "urinary_bladder", "sacrum"]),
    ("Thigh", ["femur_"]),
]
regions = []
for title, keys in REGIONS:
    raws = []
    for k in keys:
        raws += [v for v in names.values() if v == k or (k.endswith("_") and v.startswith(k)) or v.startswith(k)]
    r = zrange(set(raws))
    if r:
        regions.append({"name": title, "k0": r[0], "k1": r[1]})

# ---- payloads ----------------------------------------------------------------
files = {}
files["ct"] = write_payload(OUT, "ct", ct_to_u8(hu).tobytes())
files["labels"] = write_payload(OUT, "labels", ts.tobytes())
mesh_ids = [s["id"] for s in structs if s["group"] != "Muscles" or True]
raw_mesh, nm, ntri = meshes_payload(ts, SPACING, mesh_ids)
files["meshes"] = write_payload(OUT, "meshes", raw_mesh)
print("meshes", nm, "triangles", ntri)

meta = {
    "kind": "ct", "dims": list(shape[::-1]), "spacing": list(SPACING),
    "origin": list(ref.GetOrigin()), "ct_table": ct_decode_table(),
    "files": files, "structures": structs, "groups": GROUP_ORDER, "regions": regions,
}

# ---- X-ray (digitally reconstructed radiographs) for the anatomy study ------
if CASE == "neg":
    # drop the table and blankets: keep only large connected tissue, holes filled
    bm = ndimage.binary_opening(hu > -350, iterations=2)
    lb, nb = ndimage.label(bm)
    sz = ndimage.sum(bm, lb, range(1, nb + 1))
    keep = np.zeros(nb + 1, bool); keep[1:] = sz > sz.max() * 0.02
    bm = keep[lb]
    bm = np.stack([ndimage.binary_fill_holes(x) for x in bm])
    mu = np.clip((hu + 1000.0) / 1000.0, 0, None) * bm
    mu = np.where(hu > 200, mu * 1.8, mu)  # exaggerate bone, like a low-kV film
    XR_GROUPS = {"Thorax", "Heart and vessels", "Abdomen", "Pelvis", "Spine", "Ribs and thoracic cage",
                 "Upper limb girdle", "Pelvis and lower limb bones"}
    xr_ids = [s["id"] for s in structs if s["group"] in XR_GROUPS]
    xr = {}
    for view, axis in (("ap", 1), ("lat", 2)):
        path = mu.sum(axis) * (SPACING[1] if axis == 1 else SPACING[0])
        img = 1 - np.exp(-path / 180.0)
        img = (img - np.percentile(img, 1)) / (np.percentile(img, 99.8) - np.percentile(img, 1))
        img = np.clip(img, 0, 1) ** 0.85
        # label: structure with the longest path along each ray
        best = np.zeros(img.shape, np.uint8); bestlen = np.zeros(img.shape, np.float32)
        for sid in xr_ids:
            ln = (ts == sid).sum(axis).astype(np.float32)
            upd = ln > bestlen
            best[upd] = sid; bestlen[upd] = ln[upd]
        best[bestlen < 3] = 0
        xr[view] = {"w": img.shape[1], "h": img.shape[0]}
        files[f"xr_{view}"] = write_payload(OUT, f"xr_{view}", (img * 255).astype(np.uint8).tobytes())
        files[f"xr_{view}_lab"] = write_payload(OUT, f"xr_{view}_lab", best.tobytes())
    meta["xray"] = xr

# ---- PET and lesions for pathology cases ------------------------------------
if CASE in ("lym", "lung"):
    pt = sitk.ReadImage(f"{ROOT}/nii/{CASE}_pt.nii.gz")
    pfile = sorted(glob.glob(f"{ROOT}/dicom/{CASE}/PT_*/*"))[0]
    p = pydicom.dcmread(pfile, stop_before_pixels=True)
    ri = p.RadiopharmaceuticalInformationSequence[0]
    t = lambda s: datetime.strptime(str(s).split(".")[0][:6], "%H%M%S")
    dt = (t(p.SeriesTime) - t(ri.RadiopharmaceuticalStartTime)).total_seconds()
    dose = float(ri.RadionuclideTotalDose) * 2 ** (-dt / float(ri.RadionuclideHalfLife))
    suv_scale = float(p.PatientWeight) * 1000.0 / dose
    suv = resample(sitk.Cast(pt, sitk.sitkFloat32), sitk.sitkLinear, 0) * suv_scale
    files["pet"] = write_payload(OUT, "pet", np.clip(np.rint(suv * 20), 0, 255).astype(np.uint8).tobytes())
    meta["pet_scale"] = 0.05  # SUV per stored step

    # lesion SEG is defined on the PET grid: rebuild it as an image with PET geometry
    sfile = glob.glob(f"{ROOT}/dicom/{CASE}/SEG_*/*")[0]
    sd = pydicom.dcmread(sfile)
    pt_arr = seg_frames_to_grid(sd, pt)
    les_img = sitk.GetImageFromArray(pt_arr); les_img.CopyInformation(pt)
    # components and their stats on the native PET grid (true SUVmax, PET voxel volume)
    suv_native = sitk.GetArrayFromImage(pt).astype(np.float32) * suv_scale
    comp_n, ncomp = ndimage.label(pt_arr > 0)
    pt_vox_ml = float(np.prod(pt.GetSpacing())) / 1000
    comp_img = sitk.GetImageFromArray(comp_n.astype(np.uint16)); comp_img.CopyInformation(pt)
    comp = resample(comp_img, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    lesions = []
    for c in range(1, ncomp + 1):
        mn = comp_n == c
        m = comp == c
        if mn.sum() < 2 or m.sum() == 0:
            continue
        organ_votes = np.bincount(ts[m], minlength=256); organ_votes[0] = 0
        if organ_votes.max() == 0:
            dil = ndimage.binary_dilation(m, iterations=3)
            organ_votes = np.bincount(ts[dil], minlength=256); organ_votes[0] = 0
        site = describe(names[int(organ_votes.argmax())]) if organ_votes.max() else None
        pts = np.argwhere(m)
        peak = pts[np.argmax(suv[m])]
        lesions.append({"ml": round(float(mn.sum()) * pt_vox_ml, 1), "suvmax": round(float(suv_native[mn].max()), 1),
                        "center": [int(peak[2]), int(peak[1]), int(peak[0])],
                        "near": (site["name"] + ("" if site["hemi"] == "M" else f" ({'left' if site['hemi']=='L' else 'right'})")) if site else None,
                        "_mask_id": c})
    lesions.sort(key=lambda l: -l["ml"])
    les_lab = np.zeros(shape, np.uint8)
    for i, l in enumerate(lesions[:60], start=1):
        les_lab[comp == l.pop("_mask_id")] = i
        l["id"] = i
    for l in lesions[60:]:
        les_lab[comp == l.pop("_mask_id")] = 61
    files["lesions"] = write_payload(OUT, "lesions", les_lab.tobytes())
    raw_mesh, nm, ntri = meshes_payload(les_lab, SPACING, list(range(1, min(len(lesions), 60) + 1)), sigma=0.6)
    files["lesion_meshes"] = write_payload(OUT, "lesion_meshes", raw_mesh)
    meta["lesions"] = lesions[:60]
    meta["lesion_summary"] = {"count": len(lesions), "total_ml": round(sum(l["ml"] for l in lesions), 1),
                              "suvmax": max(l["suvmax"] for l in lesions) if lesions else None}
    meta["patient"] = {"age": str(p.get("PatientAge", "")), "sex": str(p.get("PatientSex", "")),
                       "weight_kg": float(p.PatientWeight)}
    print("lesions", len(lesions), meta["lesion_summary"])

write_meta(OUT, meta)
tot = sum(v["png_bytes"] for v in files.values())
print("payload MB", round(tot / 1e6, 1), {k: round(v["png_bytes"] / 1e6, 2) for k, v in files.items()})
