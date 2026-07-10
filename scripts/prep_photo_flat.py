"""
Prepare a FLAT / ILLUSTRATED image (pixel art, cartoon, solid-ish background)
for ASCII conversion. prep_photo.py uses rembg's AI cutout, which works on
photos but can eat subject props whose color resembles the background (e.g. a
gray phone on a beige wall). This variant removes the background by color
instead:

  1. estimate the background color from the top rows of the image
  2. mark pixels near that color, then keep only the connected regions that
     touch the image border -- interior areas of a similar color (face
     highlights, a held object) survive
  3. boost LOCAL contrast (CLAHE) and composite onto pure white, same as
     prep_photo.py, then crop to the ASCII grid aspect (1024x1152)

Output: source-prepped.png (grayscale), consumed by make_ascii_svg.py.

    python scripts/prep_photo_flat.py <input.png> [output.png]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

BG_SAMPLE_ROWS = 40      # rows at the top assumed to be pure background
COLOR_TOLERANCE = 45.0   # max RGB distance to count a pixel as background-colored
BLUR_SIGMA = 2.5         # smooths halftone dots before color matching
CROP_W, CROP_H = 1024, 1024  # square logo source; matches make_ascii_svg.py's 100x53 grid aspect

rgb = np.array(Image.open(INP).convert("RGB"))

# 1. background color = median of the top rows (robust to halftone noise)
bg_color = np.median(rgb[:BG_SAMPLE_ROWS].reshape(-1, 3), axis=0)

# 2. candidate background pixels by color distance, on a blurred copy
blurred = cv2.GaussianBlur(rgb, (0, 0), BLUR_SIGMA).astype(np.float32)
dist = np.linalg.norm(blurred - bg_color, axis=2)
candidate = (dist < COLOR_TOLERANCE).astype(np.uint8)

# keep only candidate regions connected to the border -- interior lookalikes
# (face highlights, held objects) are NOT background
n_labels, labels = cv2.connectedComponents(candidate, connectivity=4)
border_labels = set(np.concatenate([
    labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
]).tolist()) - {0}
background = np.isin(labels, list(border_labels)) & (candidate == 1)

# 3. grayscale + CLAHE local contrast, light global lift (as prep_photo.py)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
gray = clahe.apply(gray)
gray = cv2.convertScaleAbs(gray, alpha=1.05, beta=18)

# composite subject onto white via a feathered mask
mask = (~background).astype(np.float32)
mask = cv2.GaussianBlur(mask, (0, 0), 1.0)
out = gray.astype(np.float32) * mask + 255.0 * (1.0 - mask)
out = np.clip(out, 0, 255).astype(np.uint8)

# crop to the ASCII grid aspect (top-anchored: keeps head + phone, trims torso)
out = out[:CROP_H, :CROP_W]

Image.fromarray(out, mode="L").save(OUT)
print("wrote", OUT, out.shape, f"(background={background.mean():.0%} of frame)")
