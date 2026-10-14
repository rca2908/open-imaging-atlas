"""Readable names, groups, colours and short notes for TotalSegmentator 'total' classes."""
import colorsys

ORD = {1: "1st", 2: "2nd", 3: "3rd"}
def ordinal(n): return ORD.get(n, f"{n}th")

GROUP_ORDER = ["Thorax", "Heart and vessels", "Abdomen", "Pelvis", "Spine", "Ribs and thoracic cage",
               "Upper limb girdle", "Pelvis and lower limb bones", "Muscles", "Head and neck", "Nervous system"]

BASE = {
    "spleen": ("Spleen", "Abdomen", (157, 108, 162), "Largest lymphoid organ; left upper quadrant under ribs 9 to 11."),
    "kidney": ("Kidney", "Abdomen", (185, 102, 83), "Retroperitoneal. The right kidney usually sits slightly lower because of the liver."),
    "gallbladder": ("Gallbladder", "Abdomen", (70, 160, 70), "Stores bile; lies in the gallbladder fossa on the inferior liver surface."),
    "liver": ("Liver", "Abdomen", (221, 130, 101), "Largest solid organ. Dual blood supply: portal vein (about 75%) and hepatic artery."),
    "stomach": ("Stomach", "Abdomen", (216, 160, 115), "Fundus, body and antrum; the gastro-oesophageal junction lies left of midline at T11."),
    "pancreas": ("Pancreas", "Abdomen", (249, 180, 111), "Retroperitoneal. Head in the duodenal C-loop, tail towards the splenic hilum."),
    "adrenal_gland": ("Adrenal gland", "Abdomen", (220, 210, 90), "Right adrenal is triangular, left is crescent-shaped; both lie above the kidneys."),
    "lung_upper_lobe": ("Upper lobe", "Thorax", (102, 168, 206), None),
    "lung_middle_lobe": ("Middle lobe", "Thorax", (88, 140, 200), "Only on the right; bounded by the horizontal and oblique fissures."),
    "lung_lower_lobe": ("Lower lobe", "Thorax", (60, 110, 180), None),
    "esophagus": ("Oesophagus", "Thorax", (211, 171, 143), "Runs behind the trachea and left atrium; passes the diaphragm at T10."),
    "trachea": ("Trachea", "Thorax", (182, 228, 255), "Divides at the carina, about the level of T4/T5."),
    "thyroid_gland": ("Thyroid gland", "Head and neck", (220, 120, 180), "Lobes either side of the trachea, joined by the isthmus at rings 2 to 4."),
    "small_bowel": ("Small bowel", "Abdomen", (205, 167, 142), "Jejunum mainly left upper, ileum mainly right lower abdomen."),
    "duodenum": ("Duodenum", "Abdomen", (255, 210, 150), "C-shaped, mostly retroperitoneal; wraps the head of the pancreas."),
    "colon": ("Colon", "Abdomen", (204, 168, 120), "Caecum, ascending, transverse, descending and sigmoid colon."),
    "urinary_bladder": ("Urinary bladder", "Pelvis", (222, 198, 50), "Extraperitoneal; its size on CT depends on filling."),
    "prostate": ("Prostate", "Pelvis", (230, 158, 140), "Below the bladder, surrounding the proximal urethra."),
    "kidney_cyst": ("Kidney cyst", "Abdomen", (240, 240, 120), "Fluid-density lesion of the kidney."),
    "sacrum": ("Sacrum", "Pelvis and lower limb bones", (230, 222, 196), None),
    "heart": ("Heart", "Heart and vessels", (206, 62, 78), "Apex points left, anterior and inferior; right ventricle is the most anterior chamber."),
    "aorta": ("Aorta", "Heart and vessels", (224, 40, 40), "Ascending aorta, arch (with three branches) and descending thoracic and abdominal aorta."),
    "pulmonary_vein": ("Pulmonary veins", "Heart and vessels", (60, 90, 220), "Return oxygenated blood to the left atrium."),
    "brachiocephalic_trunk": ("Brachiocephalic trunk", "Heart and vessels", (236, 90, 60), "First branch of the aortic arch; splits into right subclavian and right common carotid."),
    "subclavian_artery": ("Subclavian artery", "Heart and vessels", (240, 110, 80), None),
    "common_carotid_artery": ("Common carotid artery", "Heart and vessels", (250, 130, 90), "Bifurcates at about C4 (upper border of the thyroid cartilage)."),
    "brachiocephalic_vein": ("Brachiocephalic vein", "Heart and vessels", (80, 110, 230), "The left vein crosses in front of the arch branches to join the right."),
    "atrial_appendage": ("Left atrial appendage", "Heart and vessels", (150, 60, 180), "Most common site of thrombus in atrial fibrillation."),
    "superior_vena_cava": ("Superior vena cava", "Heart and vessels", (70, 100, 240), "Drains into the right atrium."),
    "inferior_vena_cava": ("Inferior vena cava", "Heart and vessels", (60, 90, 210), "Right of the aorta; passes the diaphragm at T8."),
    "portal_vein_and_splenic_vein": ("Portal and splenic veins", "Heart and vessels", (0, 150, 200), "Portal vein forms behind the pancreatic neck from the splenic and superior mesenteric veins."),
    "iliac_artery": ("Common iliac artery", "Heart and vessels", (230, 70, 70), None),
    "iliac_vena": ("Common iliac vein", "Heart and vessels", (90, 120, 230), None),
    "humerus": ("Humerus", "Upper limb girdle", (230, 222, 196), None),
    "scapula": ("Scapula", "Upper limb girdle", (225, 212, 180), None),
    "clavicula": ("Clavicle", "Upper limb girdle", (235, 225, 195), "Only long bone that ossifies by membranous ossification."),
    "femur": ("Femur", "Pelvis and lower limb bones", (230, 222, 196), None),
    "hip": ("Hip bone", "Pelvis and lower limb bones", (225, 215, 185), "Ilium, ischium and pubis fused at the acetabulum."),
    "spinal_cord": ("Spinal cord", "Nervous system", (255, 230, 120), "Ends at the conus medullaris, around L1/L2 in adults."),
    "gluteus_maximus": ("Gluteus maximus", "Muscles", (170, 80, 70), None),
    "gluteus_medius": ("Gluteus medius", "Muscles", (180, 95, 80), None),
    "gluteus_minimus": ("Gluteus minimus", "Muscles", (190, 110, 90), None),
    "autochthon": ("Erector spinae", "Muscles", (160, 90, 90), "Deep (autochthonous) back muscles along the spine."),
    "iliopsoas": ("Iliopsoas", "Muscles", (200, 100, 95), "Psoas major and iliacus; the main hip flexor."),
    "brain": ("Brain", "Nervous system", (240, 180, 200), None),
    "skull": ("Skull", "Head and neck", (235, 225, 200), None),
    "sternum": ("Sternum", "Ribs and thoracic cage", (230, 220, 190), "Manubrium, body and xiphoid; the sternal angle marks the 2nd rib."),
    "costal_cartilages": ("Costal cartilages", "Ribs and thoracic cage", (200, 210, 220), None),
}

LUNG_NOTE = {
    "upper": "Above the oblique fissure (left) or horizontal fissure (right).",
    "lower": "Below and behind the oblique fissure; the largest lobe.",
}


def vert_color(level):
    region = level[0]
    hue = {"C": 0.12, "T": 0.08, "L": 0.05, "S": 0.03}[region]
    n = int(level[1:])
    light = 0.55 + 0.25 * ((n % 2))
    r, g, b = colorsys.hls_to_rgb(hue, light, 0.55)
    return (int(r * 255), int(g * 255), int(b * 255))


def describe(raw):
    """raw TotalSegmentator name -> dict(name, group, hemi, color, note)."""
    hemi = "M"
    key = raw
    for side, h in (("_left", "L"), ("_right", "R")):
        if raw.endswith(side):
            hemi, key = h, raw[: -len(side)]
    if raw.startswith("vertebrae_"):
        lvl = raw.split("_")[1]
        name = "Sacral vertebra S1" if lvl == "S1" else f"{lvl} vertebra"
        note = {"C1": "Atlas: no body, articulates with the occipital condyles.",
                "C2": "Axis: the dens projects up into the atlas.",
                "C7": "Vertebra prominens: longest spinous process in the neck.",
                "T12": "Last rib-bearing vertebra.", "L5": "Largest vertebral body."}.get(lvl)
        return dict(name=name, group="Spine", hemi="M", color=vert_color(lvl), note=note, order=lvl)
    if raw.startswith("rib_"):
        _, side, n = raw.split("_")
        n = int(n)
        note = "Floating rib: no anterior attachment." if n >= 11 else ("False rib: joins the costal margin." if n >= 8 else None)
        c = (225, 205, 170) if n % 2 else (205, 190, 160)
        return dict(name=f"{ordinal(n)} rib", group="Ribs and thoracic cage", hemi="L" if side == "left" else "R",
                    color=c, note=note, order=n)
    for lobe in ("upper", "middle", "lower"):
        if raw.startswith(f"lung_{lobe}_lobe"):
            name, group, col, note = BASE[f"lung_{lobe}_lobe"]
            hemi = "L" if raw.endswith("left") else "R"
            return dict(name=f"Lung, {name.lower()}", group=group, hemi=hemi, color=col,
                        note=note or LUNG_NOTE.get(lobe))
    if key == "atrial_appendage":
        hemi = "M"
    name, group, col, note = BASE[key]
    return dict(name=name, group=group, hemi=hemi, color=col, note=note)
