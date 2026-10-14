"""Crop each CT to the body and resample to the atlas grid (2 x 2 x 2.5 mm)."""
import sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

SPACING = (2.0, 2.0, 2.5)
for case in sys.argv[1:]:
    ct = sitk.ReadImage(f"nii/{case}_ct.nii.gz")
    a = sitk.GetArrayFromImage(ct)
    body = ndimage.binary_opening(a > -350, iterations=2)
    lab, n = ndimage.label(body)
    sizes = ndimage.sum(body, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = sizes > sizes.max() * 0.05
    nz = np.argwhere(keep[lab])
    lo = np.maximum(nz.min(0) - [0, 6, 6], 0); hi = np.minimum(nz.max(0) + 1 + [0, 6, 6], a.shape)
    del a, lab, body
    roi = sitk.RegionOfInterest(ct, [int(hi[2]-lo[2]), int(hi[1]-lo[1]), int(hi[0]-lo[0])], [int(lo[2]), int(lo[1]), int(lo[0])])
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(roi.GetSize(), roi.GetSpacing(), SPACING)]
    sm = sitk.SmoothingRecursiveGaussian(sitk.Cast(roi, sitk.sitkFloat32), [0.6, 0.6, 0.01])
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing(SPACING); f.SetSize(size); f.SetOutputOrigin(roi.GetOrigin()); f.SetOutputDirection(roi.GetDirection())
    f.SetInterpolator(sitk.sitkLinear); f.SetDefaultPixelValue(-1024)
    out = sitk.Cast(sitk.Round(f.Execute(sm)), sitk.sitkInt16)
    sitk.WriteImage(out, f"nii/{case}_ct2.nii.gz")
    print(case, out.GetSize(), out.GetOrigin())
