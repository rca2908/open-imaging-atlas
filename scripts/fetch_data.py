"""Download the public source data and convert CT/PET to NIfTI.

Run from the imaging-atlas/ folder:  python scripts/fetch_data.py
Needs: idc-index, SimpleITK, curl. Writes raw/, dicom/ and nii/ (git-ignored).
"""
import glob, os, subprocess
import SimpleITK as sitk
from idc_index import IDCClient

os.makedirs("raw", exist_ok=True)
os.makedirs("nii", exist_ok=True)

# Brain: ICBM 2009c template + labels from TemplateFlow
TF = "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym"
for f in ["res-01_T1w", "res-01_T2w", "res-01_seg-aseg_dseg", "res-01_desc-head_mask",
          "res-01_desc-brain_mask", "res-02_desc-DKT31_dseg"]:
    subprocess.run(["curl", "-sSf", "-o", f"raw/{f}.nii.gz", f"{TF}_{f}.nii.gz"], check=True)

c = IDCClient()
# FDG-PET-CT-Lesions (CC BY 4.0): control, lymphoma, lung cancer
STUDIES = {
    "neg": "1.3.6.1.4.1.14519.5.2.1.4219.6651.229831639145582819625933573635",
    "lym": "1.3.6.1.4.1.14519.5.2.1.4219.6651.155503527674492131788997913495",
    "lung": "1.3.6.1.4.1.14519.5.2.1.4219.6651.252049179771587308082173294971",
}
for k, uid in STUDIES.items():
    c.download_from_selection(studyInstanceUID=uid, downloadDir=f"dicom/{k}",
                              dirTemplate="%Modality_%SeriesInstanceUID", quiet=True, show_progress_bar=False)

# Prostate-MRI-US-Biopsy (CC BY 4.0): patient 0001, T2 MRI, SEGs, US and STL models
df = c.sql_query("SELECT SeriesInstanceUID, Modality, SeriesDescription FROM index "
                 "WHERE PatientID='Prostate-MRI-US-Biopsy-0001'")
want = df[df.Modality.isin(["US", "SEG", "M3D"]) | df.SeriesDescription.fillna("").str.contains("t2", case=False)]
c.download_from_selection(seriesInstanceUID=list(want.SeriesInstanceUID), downloadDir="dicom/prostate",
                          dirTemplate="%Modality_%SeriesInstanceUID", quiet=True, show_progress_bar=False)

# CT and PET series -> NIfTI
for case in STUDIES:
    for mod in ("CT", "PT"):
        d = glob.glob(f"dicom/{case}/{mod}_*")[0]
        r = sitk.ImageSeriesReader(); r.SetFileNames(r.GetGDCMSeriesFileNames(d))
        sitk.WriteImage(r.Execute(), f"nii/{case}_{mod.lower()}.nii.gz")
print("done")
