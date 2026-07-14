import cv2
import numpy as np
import os

def buat_bentuk(shape, size=300):
    """Membuat gambar bentuk geometris"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img.fill(255)  # Background putih
    
    center = (size//2, size//2)
    radius = size//3
    
    if shape == 'lingkaran':
        cv2.circle(img, center, radius, (0, 0, 255), -1)
    elif shape == 'persegi':
        pt1 = (center[0]-radius, center[1]-radius)
        pt2 = (center[0]+radius, center[1]+radius)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), -1)
    elif shape == 'segitiga':
        pts = np.array([
            [center[0], center[1]-radius],
            [center[0]-radius, center[1]+radius],
            [center[0]+radius, center[1]+radius]
        ], np.int32)
        cv2.fillPoly(img, [pts], (255, 0, 0))
    elif shape == 'bintang':
        # Bintang sederhana
        pts = []
        for i in range(10):
            angle = i * 36 * np.pi / 180
            r = radius if i % 2 == 0 else radius * 0.4
            x = center[0] + r