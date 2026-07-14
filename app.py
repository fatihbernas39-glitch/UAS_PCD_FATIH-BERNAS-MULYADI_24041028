import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import Label, Button, Frame
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

from preprocess import Preprocess
from segmentation import Segmentasi
from feature_extraction import EkstraksiFitur
from classifier import Klasifikasi

class AplikasiPCD:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Computer Vision - Praproses, Segmentasi & Pengenalan Pola")
        self.root.geometry("1400x800")
        
        # Variabel untuk menyimpan citra
        self.citra_asli = None
        self.citra_processed = None
        self.citra_segmented = None
        self.citra_path = None
        
        # Inisialisasi objek fungsi
        self.preprocess = Preprocess()
        self.segmentasi = Segmentasi()
        self.ekstraksi = EkstraksiFitur()
        self.klasifikasi = Klasifikasi()
        
        # Coba train k-NN jika dataset ada
        self.dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
        if os.path.exists(self.dataset_path):
            try:
                self.klasifikasi.train_knn(self.dataset_path, self.ekstraksi)
                print("k-NN berhasil di-train!")
            except Exception as e:
                print(f"k-NN train gagal: {e}")
        
        # Buat GUI
        self.buat_gui()
        
    def buat_gui(self):
        # ========== FRAME KIRI: Kontrol ==========
        frame_kiri = Frame(self.root, width=350, bg='#f0f0f0')
        frame_kiri.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Judul
        Label(frame_kiri, text="CONTROL PANEL", font=("Arial", 16, "bold"), 
              bg='#f0f0f0').pack(pady=10)
        
        # ========== 1. LOAD CITRA ==========
        Label(frame_kiri, text="📂 LOAD CITRA", font=("Arial", 13, "bold"),
              bg='#f0f0f0').pack(pady=(10,5))
        
        btn_load = Button(frame_kiri, text="Pilih Citra", command=self.load_image,
                         bg='#4CAF50', fg='white', font=("Arial", 11), height=1)
        btn_load.pack(pady=5, fill=tk.X, padx=10)
        
        # ========== 2. PRAPROSES ==========
        Label(frame_kiri, text="🔧 PRAPROSES (Pilih 1 atau lebih)", font=("Arial", 13, "bold"),
              bg='#f0f0f0').pack(pady=(15,5))
        
        self.pilihan_praproses = {}
        teknik = [
            ("Blur (Gaussian)", "blur"),
            ("Sharpening", "sharpening"),
            ("Edge Detection (Canny)", "edge"),
            ("Denoising", "denoising"),
            ("Morfologi (Opening)", "morfologi"),
            ("Adaptive Threshold", "adaptive"),
            ("Histogram Equalization", "histogram")
        ]
        
        for text, key in teknik:
            var = tk.BooleanVar()
            self.pilihan_praproses[key] = var
            tk.Checkbutton(frame_kiri, text=text, variable=var, bg='#f0f0f0',
                          font=("Arial", 10), anchor='w').pack(fill=tk.X, padx=20)
        
        btn_proses = Button(frame_kiri, text="▶ Jalankan Praproses", 
                           command=self.run_preprocess,
                           bg='#2196F3', fg='white', font=("Arial", 11))
        btn_proses.pack(pady=5, fill=tk.X, padx=10)
        
        # ========== 3. SEGMENTASI ==========
        Label(frame_kiri, text="✂️ SEGMENTASI", font=("Arial", 13, "bold"),
              bg='#f0f0f0').pack(pady=(15,5))
        
        self.metode_seg = tk.StringVar(value="threshold")
        tk.Radiobutton(frame_kiri, text="Thresholding", variable=self.metode_seg, 
                      value="threshold", bg='#f0f0f0', anchor='w').pack(fill=tk.X, padx=20)
        tk.Radiobutton(frame_kiri, text="Watershed", variable=self.metode_seg,
                      value="watershed", bg='#f0f0f0', anchor='w').pack(fill=tk.X, padx=20)
        tk.Radiobutton(frame_kiri, text="K-Means", variable=self.metode_seg,
                      value="kmeans", bg='#f0f0f0', anchor='w').pack(fill=tk.X, padx=20)
        
        btn_seg = Button(frame_kiri, text="▶ Jalankan Segmentasi", 
                        command=self.run_segmentation,
                        bg='#FF9800', fg='white', font=("Arial", 11))
        btn_seg.pack(pady=5, fill=tk.X, padx=10)
        
        # ========== 4. EKSTRAKSI FITUR ==========
        Label(frame_kiri, text="📊 EKSTRAKSI FITUR", font=("Arial", 13, "bold"),
              bg='#f0f0f0').pack(pady=(15,5))
        
        btn_ekstrak = Button(frame_kiri, text="▶ Ekstrak Fitur", 
                            command=self.run_feature_extraction,
                            bg='#9C27B0', fg='white', font=("Arial", 11))
        btn_ekstrak.pack(pady=5, fill=tk.X, padx=10)
        
        # ========== 5. PENGENALAN POLA ==========
        Label(frame_kiri, text="🎯 PENGENALAN POLA", font=("Arial", 13, "bold"),
              bg='#f0f0f0').pack(pady=(15,5))
        
        self.metode_klas = tk.StringVar(value="rule")
        tk.Radiobutton(frame_kiri, text="Rule-Based", variable=self.metode_klas,
                      value="rule", bg='#f0f0f0', anchor='w').pack(fill=tk.X, padx=20)
        tk.Radiobutton(frame_kiri, text="k-NN", variable=self.metode_klas,
                      value="knn", bg='#f0f0f0', anchor='w').pack(fill=tk.X, padx=20)
        
        btn_klas = Button(frame_kiri, text="▶ Klasifikasi / Pengenalan", 
                         command=self.run_classification,
                         bg='#E91E63', fg='white', font=("Arial", 11))
        btn_klas.pack(pady=5, fill=tk.X, padx=10)
        
        # ========== 6. RESET ==========
        btn_reset = Button(frame_kiri, text="🔄 Reset Tampilan", 
                          command=self.reset_display,
                          bg='#9E9E9E', fg='white', font=("Arial", 11))
        btn_reset.pack(pady=10, fill=tk.X, padx=10)
        
        # ========== FRAME KANAN: Tampilan Citra ==========
        frame_kanan = Frame(self.root)
        frame_kanan.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label untuk menampilkan citra
        self.canvas_asli = Label(frame_kanan, text="Citra Asli", 
                                bg='#e0e0e0', width=50, height=25)
        self.canvas_asli.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        self.canvas_hasil = Label(frame_kanan, text="Hasil Processing",
                                 bg='#e0e0e0', width=50, height=25)
        self.canvas_hasil.pack(side=tk.RIGHT, padx=5, fill=tk.BOTH, expand=True)
        
        # Label informasi
        self.label_info = Label(frame_kiri, text="Status: Siap", bg='#f0f0f0', 
                               font=("Arial", 9), wraplength=300)
        self.label_info.pack(pady=5)
        
        # Hasil ekstraksi fitur
        self.frame_fitur = Frame(frame_kiri, bg='#f0f0f0')
        self.frame_fitur.pack(pady=5, fill=tk.X)
        
    def load_image(self):
        """Memuat citra dari file"""
        file_path = filedialog.askopenfilename(
            title="Pilih Citra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.citra_path = file_path
            self.citra_asli = cv2.imread(file_path)
            self.citra_asli = cv2.cvtColor(self.citra_asli, cv2.COLOR_BGR2RGB)
            self.citra_processed = self.citra_asli.copy()
            self.citra_segmented = None
            
            self.tampilkan_citra(self.citra_asli, self.canvas_asli)
            self.tampilkan_citra(self.citra_processed, self.canvas_hasil)
            
            self.label_info.config(text=f"Citra dimuat: {os.path.basename(file_path)}")
    
    def tampilkan_citra(self, citra, canvas, ukuran=(500, 450)):
        """Menampilkan citra di canvas"""
        if citra is None:
            return
        
        # Jika citra grayscale, konversi ke RGB
        if len(citra.shape) == 2:
            citra = cv2.cvtColor(citra, cv2.COLOR_GRAY2RGB)
        elif len(citra.shape) == 3 and citra.shape[2] == 1:
            citra = cv2.cvtColor(citra, cv2.COLOR_GRAY2RGB)
        
        # Pastikan tipe data uint8
        if citra.dtype != np.uint8:
            citra = np.clip(citra, 0, 255).astype(np.uint8)
        
        # Resize
        h, w = citra.shape[:2]
        if h > ukuran[1] or w > ukuran[0]:
            scale = min(ukuran[0]/w, ukuran[1]/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            citra_resized = cv2.resize(citra, (new_w, new_h))
        else:
            citra_resized = citra.copy()
        
        # Konversi ke PIL
        try:
            img_pil = Image.fromarray(citra_resized.astype('uint8'))
            img_tk = ImageTk.PhotoImage(img_pil)
            canvas.config(image=img_tk)
            canvas.image = img_tk
        except Exception as e:
            print(f"Error tampilkan citra: {e}")
    
    def run_preprocess(self):
        """Menjalankan praproses sesuai pilihan"""
        if self.citra_asli is None:
            messagebox.showwarning("Peringatan", "Mohon muat citra terlebih dahulu!")
            return
        
        terpilih = [k for k, v in self.pilihan_praproses.items() if v.get()]
        
        # ========== UBAH: MINIMAL 1 TEKNIK ==========
        if len(terpilih) < 1:
            messagebox.showwarning("Peringatan", "Pilih minimal 1 teknik!")
            return
        
        print(f"\n=== PRAPROSES DIMULAI ===")
        print(f"Teknik dipilih: {terpilih}")
        
        citra = self.citra_asli.copy()
        nama_teknik = []
        
        try:
            # ========== URUTAN EKSEKUSI ==========
            
            # 1. Perbaiki kualitas gambar
            if 'histogram' in terpilih:
                print("-> Histogram Equalization...")
                citra = self.preprocess.histogram_equalization(citra)
                nama_teknik.append("Histogram")
            
            if 'denoising' in terpilih:
                print("-> Denoising...")
                citra = self.preprocess.denoising(citra)
                nama_teknik.append("Denoising")
            
            if 'blur' in terpilih:
                print("-> Gaussian Blur...")
                citra = self.preprocess.gaussian_blur(citra)
                nama_teknik.append("Blur")
            
            if 'sharpening' in terpilih:
                print("-> Sharpening...")
                citra = self.preprocess.sharpening(citra)
                nama_teknik.append("Sharpening")
            
            # 2. Deteksi tepi
            if 'edge' in terpilih:
                print("-> Edge Detection...")
                citra = self.preprocess.edge_detection(citra)
                nama_teknik.append("Edge")
            
            # 3. Morfologi
            if 'morfologi' in terpilih:
                print("-> Morfologi...")
                citra = self.preprocess.morfologi(citra)
                nama_teknik.append("Morfologi")
            
            # 4. Adaptive Threshold (terakhir)
            if 'adaptive' in terpilih:
                print("-> Adaptive Threshold...")
                citra = self.preprocess.adaptive_threshold(citra)
                nama_teknik.append("Adaptive")
            
            print(f"=== PRAPROSES SELESAI ===")
            print(f"Shape akhir: {citra.shape}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")
            return
        
        # CEK: Apakah citra valid?
        if citra is None:
            print("ERROR: Citra hasil None!")
            return
        
        # CEK: Apakah semua nilai sama (hitam atau putih)?
        if len(citra.shape) == 3:
            if np.all(citra == 0):
                print("ERROR: Semua pixel bernilai 0 (HITAM)!")
                messagebox.showwarning("Peringatan", 
                    "Hasil praproses semua hitam. Coba teknik lain!")
                return
            
            if np.all(citra == 255):
                print("ERROR: Semua pixel bernilai 255 (PUTIH)!")
                messagebox.showwarning("Peringatan", 
                    "Hasil praproses semua putih. Coba teknik lain!")
                return
        
        self.citra_processed = citra
        self.tampilkan_citra(citra, self.canvas_hasil)
        
        info = f"Praproses selesai: {', '.join(nama_teknik)}"
        self.label_info.config(text=info)
    
    def run_segmentation(self):
        """Menjalankan segmentasi"""
        if self.citra_processed is None:
            messagebox.showwarning("Peringatan", "Lakukan praproses terlebih dahulu!")
            return
        
        metode = self.metode_seg.get()
        print(f"\n=== SEGMENTASI DIMULAI ===")
        print(f"Metode: {metode}")
        
        try:
            if metode == "threshold":
                citra_seg = self.segmentasi.threshold_segmentation(self.citra_processed)
            elif metode == "watershed":
                citra_seg = self.segmentasi.watershed_segmentation(self.citra_processed)
            elif metode == "kmeans":
                citra_seg = self.segmentasi.kmeans_segmentation(self.citra_processed)
            else:
                return
        except Exception as e:
            print(f"ERROR: {e}")
            messagebox.showerror("Error", f"Segmentasi gagal: {str(e)}")
            return
        
        self.citra_segmented = citra_seg
        self.tampilkan_citra(citra_seg, self.canvas_hasil)
        
        metode_nama = {"threshold": "Thresholding", "watershed": "Watershed", "kmeans": "K-Means"}
        self.label_info.config(text=f"Segmentasi selesai: {metode_nama.get(metode, metode)}")
        print(f"=== SEGMENTASI SELESAI ===")
    
    def run_feature_extraction(self):
        """Ekstraksi fitur dari hasil segmentasi"""
        if self.citra_segmented is None:
            messagebox.showwarning("Peringatan", "Lakukan segmentasi terlebih dahulu!")
            return
        
        print(f"\n=== EKSTRAKSI FITUR DIMULAI ===")
        
        try:
            fitur = self.ekstraksi.ekstrak_semua_fitur(self.citra_segmented)
        except Exception as e:
            print(f"ERROR: {e}")
            messagebox.showerror("Error", f"Ekstraksi fitur gagal: {str(e)}")
            return
        
        # Tampilkan hasil fitur
        for widget in self.frame_fitur.winfo_children():
            widget.destroy()
        
        teks = "📊 HASIL EKSTRAKSI FITUR:\n"
        teks += f"  Luas: {fitur['luas']:.2f}\n"
        teks += f"  Keliling: {fitur['keliling']:.2f}\n"
        teks += f"  Circularity: {fitur['circularity']:.3f}\n"
        teks += f"  Mean R: {fitur['mean_r']:.2f}\n"
        teks += f"  Mean G: {fitur['mean_g']:.2f}\n"
        teks += f"  Mean B: {fitur['mean_b']:.2f}"
        
        Label(self.frame_fitur, text=teks, bg='#f0f0f0', font=("Courier", 9),
              justify=tk.LEFT).pack()
        
        self.label_info.config(text="Ekstraksi fitur selesai")
        print(f"=== EKSTRAKSI FITUR SELESAI ===")
    
    def run_classification(self):
        """Pengenalan pola/klasifikasi"""
        if self.citra_segmented is None:
            messagebox.showwarning("Peringatan", "Lakukan segmentasi terlebih dahulu!")
            return
        
        metode = self.metode_klas.get()
        print(f"\n=== KLASIFIKASI DIMULAI ===")
        print(f"Metode: {metode}")
        
        try:
            fitur = self.ekstraksi.ekstrak_semua_fitur(self.citra_segmented)
            
            if metode == "rule":
                hasil = self.klasifikasi.classify_rule_based(fitur)
            elif metode == "knn":
                hasil = self.klasifikasi.classify_knn(fitur)
            else:
                hasil = "Metode tidak dikenal"
        except Exception as e:
            print(f"ERROR: {e}")
            messagebox.showerror("Error", f"Klasifikasi gagal: {str(e)}")
            return
        
        messagebox.showinfo("Hasil Pengenalan", 
                           f"🔍 Hasil Klasifikasi:\n\nKelas: {hasil}\n\nMetode: {metode}")
        self.label_info.config(text=f"Hasil: {hasil}")
        print(f"Hasil: {hasil}")
        print(f"=== KLASIFIKASI SELESAI ===")
    
    def reset_display(self):
        """Reset tampilan ke citra asli"""
        if self.citra_asli is not None:
            self.citra_processed = self.citra_asli.copy()
            self.citra_segmented = None
            self.tampilkan_citra(self.citra_asli, self.canvas_hasil)
            self.label_info.config(text="Tampilan di-reset")
            
            for widget in self.frame_fitur.winfo_children():
                widget.destroy()

# ========== MAIN ==========
if __name__ == "__main__":
    root = tk.Tk()
    app = AplikasiPCD(root)
    root.mainloop()