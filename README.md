# Jejak Platform

Deteksi asal platform (O, WA, TG, IG, X) pada foto lewat DFT 2D + fitur spasial.

## Jalankan
    pip install -r requirements.txt
    streamlit run app.py

## Data
    data/obj01/O.jpg WA.jpg TG.jpg IG.jpg X.jpg
    data/obj02/...
Nama folder = ID objek (untuk GroupKFold), nama file = label platform.
Tombol "Buat data demo" di langkah 1 membuat data sintetis hanya untuk mencoba aplikasi.

## File
- fitur.py : pra-proses, 57 fitur, evaluasi (bisa dipakai tanpa UI)
- app.py   : antarmuka Streamlit (Dataset > Ekstraksi > Evaluasi > Uji foto)
