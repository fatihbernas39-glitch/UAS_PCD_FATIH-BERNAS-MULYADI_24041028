import cv2
import numpy as np

class Segmentasi:
    def __init__(self):
        pass
    
    def threshold_segmentation(self, citra):
        """Segmentasi dengan Thresholding"""
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hasil = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(hasil, contours, -1, (0, 255, 0), 2)
        
        return hasil
    
    def watershed_segmentation(self, citra):
        """Segmentasi dengan Watershed"""
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
            img_rgb = citra.copy()
        else:
            gray = citra
            img_rgb = cv2.cvtColor(citra, cv2.COLOR_GRAY2RGB)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img_rgb, markers)
        
        hasil = img_rgb.copy()
        hasil[markers == -1] = [0, 255, 0]
        
        return hasil
    
    def kmeans_segmentation(self, citra, K=2):
        """Segmentasi dengan K-Means Clustering"""
        if len(citra.shape) == 3:
            pixels = citra.reshape((-1, 3))
        else:
            pixels = citra.reshape((-1, 1))
        
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        
        if len(citra.shape) == 3:
            segmented = segmented.reshape(citra.shape)
        else:
            segmented = segmented.reshape(citra.shape)
        
        return segmented