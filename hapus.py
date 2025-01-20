import shutil
import os

# Menghapus folder "data" jika ada
if os.path.exists("data"):
    shutil.rmtree("data")
    print("Folder 'data' berhasil dihapus.")

# Menghapus file "embedding_manual.xlsx" jika ada
if os.path.exists("embedding_manual_laporan.xlsx"):
    os.remove("embedding_manual_laporan.xlsx")
    print("File 'embedding_manual_laporan.xlsx' berhasil dihapus.")
else:
    print("File 'embedding_manual_laporan.xlsx' tidak ditemukan.")
