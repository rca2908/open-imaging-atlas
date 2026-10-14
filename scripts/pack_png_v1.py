"""Re-pack the gzip .bin data files as lossless 8-bit grayscale PNGs (4096 px wide),
because artifacts only serve web file types. Byte length goes into meta.json."""
import gzip, json, sys, zlib
import numpy as np
from PIL import Image
d = sys.argv[1]
meta = json.load(open(f"{d}/meta.json"))
meta["files"] = {}
for name in ("t1", "t2", "labels", "meshes"):
    raw = gzip.decompress(open(f"{d}/{name}.bin", "rb").read())
    n = len(raw); W = 4096; H = -(-n // W)
    arr = np.zeros(W * H, np.uint8); arr[:n] = np.frombuffer(raw, np.uint8)
    Image.fromarray(arr.reshape(H, W), "L").save(f"{d}/{name}.png", optimize=True, compress_level=9)
    meta["files"][name] = {"bytes": n, "adler32": zlib.adler32(raw)}
    print(name, n)
json.dump(meta, open(f"{d}/meta.json", "w"), separators=(",", ":"))
