import cv2
import numpy as np

# Example 8x8 block
block = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 66, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
], dtype=np.float32)

# Apply DCT
dct_block = cv2.dct(block)

# Apply IDCT
idct_block = cv2.idct(dct_block)

# Check the difference
print("Original Block:\n", block)
print("Reconstructed Block:\n", idct_block)
print("Difference:\n", block - idct_block)