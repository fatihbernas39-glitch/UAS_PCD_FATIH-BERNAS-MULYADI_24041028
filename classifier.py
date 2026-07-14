import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

class Klasifikasi:
    def __init__(self):
        self.knn_model = None
        self.class_names = []
        
    def classify_rule_based(self, fitur):
        """
        Klasifikasi berdasarkan aturan (rule-based)
        Menggunakan circularity dan luas untuk membedakan bentuk
        """
        circularity = fitur['circularity']
        luas = fitur['luas']
        
        # Aturan untuk bentuk geometris
        # Nilai threshold disesuaikan dengan dataset Anda
        
        # Lingkaran: circularity mendekati 1
        if circularity > 0.85:
            return "Lingkaran"
        
        # Persegi: circularity sekitar 0.78
        elif circularity > 0.70 and circularity <= 0.85:
            return "Persegi"
        
        # Segitiga: circularity rendah
        elif circularity > 0.55 and circularity <= 0.70:
            return "Segitiga"
        
        # Bintang: circularity sangat rendah
        elif circularity <= 0.55:
            return "Bintang"
        
        else:
            return "Tidak Teridentifikasi"
    
    def _fitur_ke_vector(self, fitur):
        """Konversi dictionary fitur ke vector untuk k-NN"""
        return np.array([
            fitur['luas'],
            fitur['keliling'],
            fitur['circularity'],
            fitur['mean_r'],
            fitur['mean_g'],
            fitur['mean_b']
        ])
    
    def train_knn(self, dataset_path, feature_extractor):
        """Melatih model k-NN menggunakan dataset"""
        self.feature_extractor = feature_extractor
        X_train = []
        y_train = []
        self.class_names = []
        
        for class_name in sorted(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                self.class_names.append(class_name)
                
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            fitur = feature_extractor.ekstrak_semua_fitur(img)
                            feature_vector = self._fitur_ke_vector(fitur)
                            X_train.append(feature_vector)
                            y_train.append(class_name)
        
        if len(X_train) > 0:
            X_train = np.array(X_train)
            self.knn_model = KNeighborsClassifier(n_neighbors=3)
            self.knn_model.fit(X_train, y_train)
            return True
        return False
    
    def classify_knn(self, fitur):
        """Klasifikasi menggunakan k-NN"""
        if self.knn_model is None:
            return "Model belum dilatih"
        
        feature_vector = self._fitur_ke_vector(fitur).reshape(1, -1)
        prediksi = self.knn_model.predict(feature_vector)
        return prediksi[0]