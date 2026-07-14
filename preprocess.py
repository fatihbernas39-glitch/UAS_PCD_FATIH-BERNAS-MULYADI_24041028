import cv2
import numpy as np

class Preprocess:
    def __init__(self):
        pass
    
    def gaussian_blur(self, citra, kernel_size=5):
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if len(citra.shape) == 3:
            blurred = cv2.GaussianBlur(citra, (kernel_size, kernel_size), 0)
            return blurred
        else:
            blurred = cv2.GaussianBlur(citra, (kernel_size, kernel_size), 0)
            return cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
    
    def sharpening(self, citra, strength=1.5):
        if len(citra.shape) == 3:
            if len(citra.shape) == 3 and citra.shape[2] == 3:
                if np.all(citra[:,:,0] == citra[:,:,1]) and np.all(citra[:,:,1] == citra[:,:,2]):
                    gray = citra[:,:,0]
                else:
                    gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
            else:
                gray = citra
        else:
            gray = citra
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharp = gray - strength * laplacian
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)
    
    def edge_detection(self, citra, threshold1=50, threshold2=150):
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        edges = cv2.Canny(gray, threshold1, threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    def denoising(self, citra, h=10):
        if len(citra.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(citra, None, h, h, 7, 21)
            return denoised
        else:
            denoised = cv2.fastNlMeansDenoising(citra, None, h, 7, 21)
            return cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    
    def morfologi(self, citra, operasi='opening', kernel_size=5):
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operasi == 'opening':
            result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operasi == 'closing':
            result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operasi == 'erosion':
            result = cv2.erode(binary, kernel, iterations=1)
        elif operasi == 'dilation':
            result = cv2.dilate(binary, kernel, iterations=1)
        else:
            result = binary
        
        return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    def adaptive_threshold(self, citra, block_size=11, C=2):
        if len(citra.shape) == 3:
            gray = cv2.cvtColor(citra, cv2.COLOR_RGB2GRAY)
        else:
            gray = citra
        
        if block_size % 2 == 0:
            block_size += 1
        
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 
            block_size, C
        )
        
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    
    def histogram_equalization(self, citra):
        if len(citra.shape) == 3:
            yuv = cv2.cvtColor(citra, cv2.COLOR_RGB2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            equalized = cv2.equalizeHist(citra)
            return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)