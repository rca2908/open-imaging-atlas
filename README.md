# Open Imaging Atlas

A browser atlas of anatomy and pathology on real, openly licensed scans: whole-body CT, radiographs reconstructed from that CT, brain and prostate MRI, 3D transrectal prostate ultrasound, and PET-CT / MRI pathology cases.

Live page: https://claude.ai/artifact/E7rChupgewSuGX2dyLFsge (private until shared).

This folder holds the page and the scripts that build its data. The generated data (~58 MB of PNG-packed volumes) is not committed; rebuild it with the steps below.

## Data sources

| Module | Source | Licence |
|---|---|---|
| Whole-body CT, lymphoma and lung cancer PET-CT | FDG-PET-CT-Lesions (Gatidis et al.), via NCI Imaging Data Commons | CC BY 4.0 |
| Prostate T2 MRI, outlines, transrectal US | Prostate-MRI-US-Biopsy (Natarajan et al.), via NCI Imaging Data Commons | CC BY 4.0 |
| Brain MRI and labels | ICBM 2009c Nonlinear Asymmetric template (Fonov et al.), via TemplateFlow | Permission to use, copy, modify, distribute with notice (McConnell Brain Imaging Centre) |
| CT organ labels | Generated here with TotalSegmentator (`total`, `--fast`, 3 mm) | Automatic, not radiologist-reviewed |

## Rebuild

Run everything from this folder. Python 3.11+.

```bash
python -m venv venv && . venv/bin/activate
pip install nibabel scipy scikit-image pillow SimpleITK pydicom highdicom idc-index TotalSegmentator

python scripts/fetch_data.py                       # raw/, dicom/, nii/
python scripts/prep_ct.py neg lym lung             # crop + resample CT to 2 x 2 x 2.5 mm
TOTALSEG_HOME_DIR=$PWD/tsweights python scripts/run_totalseg.py neg lym lung   # needs ~16 GB RAM

# brain (first version's pipeline, then repacked)
mkdir -p build/brain_v1 && python scripts/build_brain_v1.py raw build/brain_v1 && python scripts/pack_png_v1.py build/brain_v1

for c in neg lym lung; do python scripts/build_ct.py $c site/data/ct_$c; done
for s in brain prostate trus; do python scripts/build_mr_us.py $s site/data/$s; done
mkdir -p site/thumbs && python scripts/thumbs.py site/data site/thumbs
```

Then serve `site/` (e.g. `python -m http.server`). The page loads three.js r128 from cdnjs.

## Notes

- Volumes are stored x-fastest in DICOM LPS index order and packed into 8-bit grayscale PNGs (4096 px wide), because the artifact host only serves web file types. `meta.json` holds byte counts and an Adler-32 checksum that the page verifies.
- CT is companded to 8 bits (fine steps from -200 to 300 HU); the page decodes it with `ct_table`.
- The lesion DICOM SEGs in FDG-PET-CT-Lesions have rows running in -y relative to the PET; `common.seg_frames_to_grid` handles SEG orientation explicitly.
- For learning only. Not for clinical use.
