import numpy as np
from PIL import Image

def angle_image(rgb, side):
    gray = np.asarray(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('L').resize((side, side), Image.Resampling.BOX), dtype=np.float32)
    return gray / max(float(gray.max()), 1.) * 2. - 1.
