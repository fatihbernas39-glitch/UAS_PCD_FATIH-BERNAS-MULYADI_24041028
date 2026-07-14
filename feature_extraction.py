import cv2
import numpy as np
import math

class EkstraksiFitur:
    def __init__(self):
        pass
    
    def ekstrak_luas(self, citra):
        """Menghitung luas objek"""
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.contourArea(largest_contour)
        return 0
    
    def ekstrak_keliling(self, citra):
        """Menghitung keliling objek"""
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.arcLength(largest_contour, True)
        return 0
    
    def ekstrak_circularity(self, citra):
        """Menghitung circularity (4π × Luas / Keliling²)"""
        luas = self.ekstrak_luas(citra)
        keliling = self.ekstrak_keliling(citra)
        
        if keliling > 0:
            circularity = (4 * math.pi * luas) / (keliling * keliling)
            return min(circularity, 1.0)
        return 0
    
    def ekstrak_mean_warna(self, citra):
        """Menghitung rata-rata warna (R, G, B)"""
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            mask = binary > 0
            
            if np.any(mask):
                mean_r = np.mean(citra[mask, 0])
                mean_g = np.mean(citra[mask, 1])
                mean_b = np.mean(citra[mask, 2])
                return mean_r, mean_g, mean_b
            return 0, 0, 0
        return 0, 0, 0
    
    def ekstrak_hu_moments(self, citra):
        """Menghitung 7 Hu Moments"""
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments)
        return hu_moments.flatten()
    
    def ekstrak_semua_fitur(self, citra):
        """Ekstrak semua fitur sekaligus"""
        fitur = {
            'luas': self.ekstrak_luas(citra),
            'keliling': self.ekstrak_keliling(citra),
            'circularity': self.ekstrak_circularity(citra),
        }
        
        mean_r, mean_g, mean_b = self.ekstrak_mean_warna(citra)
        fitur['mean_r'] = mean_r
        fitur['mean_g'] = mean_g
        fitur['mean_b'] = mean_b
        
        hu = self.ekstrak_hu_moments(citra)
        for i, val in enumerate(hu):
            fitur[f'hu_{i+1}'] = val
        
        return fitur