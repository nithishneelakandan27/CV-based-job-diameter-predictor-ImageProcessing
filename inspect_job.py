from pathlib import Path
import cv2
import numpy as np

img_path = Path("data/input/WhatsApp Image 2026-09-07 at 11.30.01 AM.jpeg")
img = cv2.imread(str(img_path))
x, y, w, h = 320, 220, 160, 480
roi = img[y:y+h, x:x+w]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
sobel = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))

# Let's inspect intensity / horizontal gradient across vertical bins or rows
for r_idx in range(0, h, 40):
    row_sobel = sobel[r_idx, :]
    max_idx = np.argmax(row_sobel)
    print(f"Row {r_idx}: max sobel x={max_idx}, val={row_sobel[max_idx]}")
