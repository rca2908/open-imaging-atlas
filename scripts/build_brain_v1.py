"""Build the data files for the MNI Brain Atlas viewer.

Source: ICBM 2009c Nonlinear Asymmetric template (MNI152NLin2009cAsym) via
TemplateFlow, 1 mm T1w/T2w, the FreeSurfer-style aseg segmentation (1 mm) and
the DKT31 cortical parcellation (2 mm). Template licence: McConnell Brain
Imaging Centre, permission to use/copy/modify/distribute with notice.

Outputs (in ./site/data):
  t1.bin, t2.bin     gzip(uint8 volume, x fastest)
  labels.bin         gzip(uint8 label volume, compact ids)
  meshes.bin         gzip(binary mesh pack, see write_meshes)
  meta.json          dims, MNI origin, structures (id, name, group, colour,
                     hemisphere, voxels, centroid)
"""
import gzip, json, struct, sys
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure

RAW, OUT = sys.argv[1], sys.argv[2]

def load(name):
    im = nib.load(f"{RAW}/{name}.nii.gz")
    return np.asarray(im.dataobj), im.affine

t1, aff = load("res-01_T1w")
t2, _ = load("res-01_T2w")
aseg, _ = load("res-01_seg-aseg_dseg")
dkt2, aff2 = load("res-02_desc-DKT31_dseg")
aseg = aseg.astype(np.int32)
dkt2 = np.rint(dkt2).astype(np.int32)

# DKT31 is 2 mm on a grid whose origin is half a voxel off the 1 mm grid.
# Map every 1 mm voxel centre to its nearest 2 mm voxel.
ii, jj, kk = np.indices(aseg.shape)
world = np.stack([ii, jj, kk], -1) @ aff[:3, :3].T + aff[:3, 3]
inv2 = np.linalg.inv(aff2)
v2 = np.rint(world @ inv2[:3, :3].T + inv2[:3, 3]).astype(int)
for a in range(3):
    np.clip(v2[..., a], 0, dkt2.shape[a] - 1, out=v2[..., a])
dkt1 = dkt2[v2[..., 0], v2[..., 1], v2[..., 2]]
del world, v2

# Cortical ribbon from aseg; parcel each ribbon voxel with the nearest DKT
# cortical label of the same hemisphere.
parcels = np.where(dkt1 >= 1000, dkt1, 0)
def nearest_fill(mask, src):
    _, idx = ndimage.distance_transform_edt(src == 0, return_indices=True)
    filled = src[idx[0], idx[1], idx[2]]
    return np.where(mask, filled, 0)
ctx = np.zeros_like(aseg)
for hemi_ctx, lo, hi in ((3, 1000, 2000), (42, 2000, 3000)):
    src = np.where((parcels >= lo) & (parcels < hi), parcels, 0)
    ctx += nearest_fill(aseg == hemi_ctx, src)

fs = aseg.copy()
fs[aseg == 77] = 0          # WM hypointensities -> dropped
fs[np.isin(aseg, (30, 62))] = 0  # vessel
fs[(aseg == 3) | (aseg == 42)] = 0
fs = np.where(ctx > 0, ctx, fs)

# --- structure table -------------------------------------------------------
SUB = {  # aseg label: (name, group, colour, hemi)
    2: ("Cerebral white matter", "White matter", (232, 228, 214), "L"),
    41: ("Cerebral white matter", "White matter", (232, 228, 214), "R"),
    251: ("Corpus callosum, splenium", "White matter", (70, 110, 210), "M"),
    252: ("Corpus callosum, posterior body", "White matter", (80, 125, 225), "M"),
    253: ("Corpus callosum, central body", "White matter", (90, 140, 240), "M"),
    254: ("Corpus callosum, anterior body", "White matter", (100, 155, 250), "M"),
    255: ("Corpus callosum, genu", "White matter", (115, 170, 255), "M"),
    85: ("Optic chiasm", "White matter", (234, 169, 30), "M"),
    4: ("Lateral ventricle", "Ventricles and CSF", (120, 18, 134), "L"),
    43: ("Lateral ventricle", "Ventricles and CSF", (120, 18, 134), "R"),
    5: ("Inferior horn of lateral ventricle", "Ventricles and CSF", (196, 58, 250), "L"),
    44: ("Inferior horn of lateral ventricle", "Ventricles and CSF", (196, 58, 250), "R"),
    14: ("Third ventricle", "Ventricles and CSF", (204, 182, 142), "M"),
    15: ("Fourth ventricle", "Ventricles and CSF", (42, 204, 164), "M"),
    24: ("Extracerebral CSF", "Ventricles and CSF", (90, 90, 96), "M"),
    31: ("Choroid plexus", "Ventricles and CSF", (0, 200, 200), "L"),
    63: ("Choroid plexus", "Ventricles and CSF", (0, 200, 200), "R"),
    10: ("Thalamus", "Deep grey matter", (0, 150, 30), "L"),
    49: ("Thalamus", "Deep grey matter", (0, 150, 30), "R"),
    11: ("Caudate nucleus", "Deep grey matter", (122, 186, 220), "L"),
    50: ("Caudate nucleus", "Deep grey matter", (122, 186, 220), "R"),
    12: ("Putamen", "Deep grey matter", (236, 13, 176), "L"),
    51: ("Putamen", "Deep grey matter", (236, 13, 176), "R"),
    13: ("Globus pallidus", "Deep grey matter", (40, 80, 255), "L"),
    52: ("Globus pallidus", "Deep grey matter", (40, 80, 255), "R"),
    26: ("Nucleus accumbens", "Deep grey matter", (255, 165, 0), "L"),
    58: ("Nucleus accumbens", "Deep grey matter", (255, 165, 0), "R"),
    17: ("Hippocampus", "Deep grey matter", (220, 216, 20), "L"),
    53: ("Hippocampus", "Deep grey matter", (220, 216, 20), "R"),
    18: ("Amygdala", "Deep grey matter", (103, 255, 255), "L"),
    54: ("Amygdala", "Deep grey matter", (103, 255, 255), "R"),
    28: ("Ventral diencephalon", "Deep grey matter", (165, 42, 42), "L"),
    60: ("Ventral diencephalon", "Deep grey matter", (165, 42, 42), "R"),
    16: ("Brainstem", "Brainstem and cerebellum", (119, 159, 176), "M"),
    7: ("Cerebellar white matter", "Brainstem and cerebellum", (220, 248, 164), "L"),
    46: ("Cerebellar white matter", "Brainstem and cerebellum", (220, 248, 164), "R"),
    8: ("Cerebellar cortex", "Brainstem and cerebellum", (230, 148, 34), "L"),
    47: ("Cerebellar cortex", "Brainstem and cerebellum", (230, 148, 34), "R"),
}
CTX = {  # DKT suffix: (name, lobe, colour)
    2: ("Caudal anterior cingulate", "Limbic", (125, 100, 160)),
    3: ("Caudal middle frontal gyrus", "Frontal", (160, 90, 40)),
    5: ("Cuneus", "Occipital", (220, 20, 100)),
    6: ("Entorhinal cortex", "Temporal", (220, 20, 10)),
    7: ("Fusiform gyrus", "Temporal", (180, 220, 140)),
    8: ("Inferior parietal lobule", "Parietal", (220, 60, 220)),
    9: ("Inferior temporal gyrus", "Temporal", (180, 40, 120)),
    10: ("Isthmus of cingulate", "Limbic", (140, 20, 140)),
    11: ("Lateral occipital cortex", "Occipital", (60, 80, 190)),
    12: ("Lateral orbitofrontal cortex", "Frontal", (55, 120, 80)),
    13: ("Lingual gyrus", "Occipital", (225, 140, 140)),
    14: ("Medial orbitofrontal cortex", "Frontal", (200, 35, 75)),
    15: ("Middle temporal gyrus", "Temporal", (160, 100, 50)),
    16: ("Parahippocampal gyrus", "Temporal", (20, 220, 60)),
    17: ("Paracentral lobule", "Frontal", (60, 220, 60)),
    18: ("Pars opercularis", "Frontal", (220, 180, 140)),
    19: ("Pars orbitalis", "Frontal", (40, 140, 80)),
    20: ("Pars triangularis", "Frontal", (220, 60, 20)),
    21: ("Pericalcarine cortex", "Occipital", (150, 125, 80)),
    22: ("Postcentral gyrus", "Parietal", (220, 20, 20)),
    23: ("Posterior cingulate", "Limbic", (220, 180, 220)),
    24: ("Precentral gyrus", "Frontal", (90, 50, 230)),
    25: ("Precuneus", "Parietal", (160, 140, 180)),
    26: ("Rostral anterior cingulate", "Limbic", (110, 40, 170)),
    27: ("Rostral middle frontal gyrus", "Frontal", (110, 80, 170)),
    28: ("Superior frontal gyrus", "Frontal", (20, 220, 160)),
    29: ("Superior parietal lobule", "Parietal", (20, 180, 140)),
    30: ("Superior temporal gyrus", "Temporal", (140, 220, 220)),
    31: ("Supramarginal gyrus", "Parietal", (80, 160, 20)),
    34: ("Transverse temporal gyrus", "Temporal", (150, 150, 200)),
    35: ("Insula", "Insular", (255, 192, 32)),
}

present = [int(v) for v in np.unique(fs) if v != 0]
structs, remap = [], np.zeros(3000, np.uint8)
for src in present:
    if src >= 1000:
        hemi = "L" if src < 2000 else "R"
        name, lobe, col = CTX[src % 1000]
        group, extra = "Cerebral cortex", {"lobe": lobe}
    else:
        name, group, col, hemi = SUB[src]
        extra = {}
    sid = len(structs) + 1
    remap[src] = sid
    structs.append({"id": sid, "src": src, "name": name, "group": group,
                    "hemi": hemi, "color": col, **extra})
labels = remap[fs]

# --- crop to the head ------------------------------------------------------
head = (t1 > np.percentile(t1, 60)) | (labels > 0)
head = ndimage.binary_opening(head, iterations=2)
nz = np.argwhere(head)
lo = np.maximum(nz.min(0) - 3, 0)
hi = np.minimum(nz.max(0) + 4, np.array(t1.shape))
sl = tuple(slice(a, b) for a, b in zip(lo, hi))
t1, t2, labels = t1[sl], t2[sl], labels[sl]
origin = (aff[:3, 3] + lo).tolist()  # MNI mm of voxel (0,0,0)

def to_u8(v):
    hi_ = np.percentile(v[v > 0], 99.7)
    return np.clip(v.astype(np.float32) / hi_ * 255, 0, 255).astype(np.uint8)

def save(name, arr):
    data = np.ascontiguousarray(arr.transpose(2, 1, 0)).tobytes()  # x fastest
    with open(f"{OUT}/{name}", "wb") as f:
        f.write(gzip.compress(data, 9))

save("t1.bin", to_u8(t1))
save("t2.bin", to_u8(t2))
save("labels.bin", labels.astype(np.uint8))

for s in structs:
    m = labels == s["id"]
    s["voxels"] = int(m.sum())
    c = np.argwhere(m).mean(0)
    # Snap the "go to" point onto the structure itself (centroids of curved
    # structures such as the caudate can fall outside them).
    pts = np.argwhere(m)
    s["center"] = pts[np.argmin(((pts - c) ** 2).sum(1))].tolist()

# --- meshes ----------------------------------------------------------------
def mesh(mask, sigma, step):
    sm = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    v, f, _, _ = measure.marching_cubes(np.pad(sm, 2), 0.5, step_size=step)
    return v - 2, f

meshes = []
SKIP_MESH = {"Cerebral white matter", "Extracerebral CSF"}
for s in structs:
    if s["group"] == "Cerebral cortex" or s["name"] in SKIP_MESH:
        continue
    big = s["voxels"] > 40000
    v, f = mesh(labels == s["id"], 0.8, 2 if big else 1)
    meshes.append((s["id"], v, f, None))

# Cerebral surface coloured by the nearest cortical parcel.
cortex_ids = np.array([s["id"] for s in structs if s["group"] == "Cerebral cortex"])
cer = np.isin(labels, cortex_ids) | np.isin(
    labels, [s["id"] for s in structs if s["group"] in ("White matter", "Deep grey matter")
             or s["name"] in ("Lateral ventricle", "Inferior horn of lateral ventricle",
                              "Third ventricle", "Choroid plexus")])
cer = ndimage.binary_closing(cer, iterations=2)
cer = ndimage.binary_fill_holes(cer)
v, f = mesh(cer, 1.2, 2)
ctx_only = np.where(np.isin(labels, cortex_ids), labels, 0)
_, idx = ndimage.distance_transform_edt(ctx_only == 0, return_indices=True)
vi = np.clip(np.rint(v).astype(int), 0, np.array(labels.shape) - 1)
vlab = ctx_only[idx[0][vi[:, 0], vi[:, 1], vi[:, 2]],
                idx[1][vi[:, 0], vi[:, 1], vi[:, 2]],
                idx[2][vi[:, 0], vi[:, 1], vi[:, 2]]]
meshes.append((0, v, f, vlab.astype(np.uint8)))

# Pack: u32 count; per mesh: u8 id, u8 hasLabels, u32 nv, u32 nf,
# nv*3 u16 (mm*64), [nv u8 labels], nf*3 u32 indices.
buf = bytearray(struct.pack("<I", len(meshes)))
tri = 0
for sid, v, f, vl in meshes:
    buf += struct.pack("<BBII", sid, vl is not None, len(v), len(f))
    buf += np.rint(v * 64).astype("<u2").tobytes()
    if vl is not None:
        buf += vl.tobytes()
    buf += f.astype("<u4").tobytes()
    tri += len(f)
with open(f"{OUT}/meshes.bin", "wb") as fh:
    fh.write(gzip.compress(bytes(buf), 9))

meta = {"dims": list(labels.shape), "origin": origin, "voxel_mm": 1,
        "structures": structs}
with open(f"{OUT}/meta.json", "w") as fh:
    json.dump(meta, fh, separators=(",", ":"))
print("dims", labels.shape, "structures", len(structs), "meshes", len(meshes), "triangles", tri)
