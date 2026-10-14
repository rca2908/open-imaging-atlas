"""Small preview images for the study cards (rendered from the packed data)."""
import json, sys, numpy as np
from PIL import Image
D = sys.argv[1]; OUT = sys.argv[2]
def meta(s): return json.load(open(f"{D}/{s}/meta.json"))
def load(s, m, n):
    f = m["files"][n]
    return np.concatenate([np.asarray(Image.open(f"{D}/{s}/{p['file']}")).reshape(-1)[:p["bytes"]] for p in f["parts"]])
def vol(s, m, n):
    nx, ny, nz = m["dims"]; return load(s, m, n).reshape(nz, ny, nx)
def win(x, c, w): return np.clip((x - (c - w / 2)) / w, 0, 1)
def colors(m):
    col = np.zeros((256, 3))
    for st in m["structures"]: col[st["id"]] = st["color"]
    return col
def over(g, l, col, a=0.45):
    rgb = np.stack([g * 255] * 3, -1)
    k = l > 0; rgb[k] = rgb[k] * (1 - a) + col[l[k]] * a
    return rgb
def save(img, name, aspect=1.0, size=360):
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    h = int(im.height * aspect)
    im = im.resize((im.width, h))
    # fit into a square tile on black
    s = size / max(im.width, im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    tile = Image.new("RGB", (size, size), (0, 0, 0)); tile.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    tile.save(f"{OUT}/{name}.png", optimize=True)

# whole-body CT: coronal soft tissue with labels, trunk only
m = meta("ct_neg"); ct = vol("ct_neg", m, "ct"); lab = vol("ct_neg", m, "labels"); tab = np.array(m["ct_table"])
reg = {r["name"]: r for r in m["regions"]}
k0, k1 = reg["Pelvis"]["k0"], reg["Neck"]["k1"]
j = m["dims"][1] // 2 + 10
sl = over(win(tab[ct[k0:k1, j, :]], 40, 400), lab[k0:k1, j, :], colors(m))[::-1]
save(sl, "ct_neg", m["spacing"][2] / m["spacing"][0])
# X-ray: chest crop of the AP DRR
xr = load("ct_neg", m, "xr_ap").reshape(m["xray"]["ap"]["h"], m["xray"]["ap"]["w"]).astype(float)
t0, t1 = reg["Thorax"]["k0"] - 10, reg["Thorax"]["k1"] + 10
c = xr[t0:t1][::-1]; lo, hi = np.percentile(c, [1, 99.5])
save(np.clip((c - lo) / (hi - lo), 0, 1) ** 1.2 * 255, "xray", m["spacing"][2] / m["spacing"][0])
# brain
m = meta("brain"); t1 = vol("brain", m, "t1"); lab = vol("brain", m, "labels")
k = 95
save(over(win(t1[k].astype(float), 147, 235), lab[k], colors(m)), "brain")
# prostate MRI
m = meta("prostate"); t2 = vol("prostate", m, "t2").astype(float); lab = vol("prostate", m, "labels")
k = m["structures"][1]["center"][2]
g = win(t2[k], 110, 220)
img = over(g, lab[k], colors(m), 0.35)[40:216, 40:216]
save(img, "prostate")
img2 = over(g, np.where(lab[k] == 2, 2, 0), colors(m), 0.55)[40:216, 40:216]
save(img2, "case_prostate")
# TRUS
m = meta("trus"); us = vol("trus", m, "us").astype(float)
save(us[:, m["dims"][1] // 2, :][::-1], "trus")
# PET MIPs (inverted, classic display)
for s in ("ct_lym", "ct_lung"):
    m = meta(s); pet = vol(s, m, "pet").astype(float) * m["pet_scale"]
    mip = pet.max(1)[::-1]
    save((1 - np.clip(mip / 6.0, 0, 1)) * 255, "case_" + s[3:], m["spacing"][2] / m["spacing"][0])
print("ok")
