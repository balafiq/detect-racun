# HIPER Apps – Pengesan Racun Berdaftar & Tidak Berdaftar

Aplikasi web (telefon / komputer) yang mengimbas botol atau bungkusan racun melalui **kamera** atau **gambar**
dan memaparkan sama ada racun itu **BERDAFTAR**, **TIDAK BERDAFTAR**, atau **TIDAK DIKENALI / TIDAK PASTI**.
Model AI berjalan sepenuhnya dalam pelayar, jadi tiada gambar dihantar ke pelayan.

Produk yang dikenali: Amistar, Antracol 70 WP, Basta 15, Dithane GR, Ecomax, Kencis, serta botol tanpa label (racun tidak berdaftar).

## Cara guna / hos

Aplikasi berada dalam folder [`docs/`](docs). Ia boleh dihoskan dengan GitHub Pages:
**Settings → Pages → Source: Deploy from a branch → Branch: `main`, folder `/docs`**.
Kamera hanya berfungsi melalui HTTPS (GitHub Pages sudah HTTPS).

Untuk mencuba secara tempatan:

```bash
python3 -m http.server 8000 --directory docs
```

Kemudian buka http://localhost:8000.

Aplikasi ini juga boleh dibenamkan dalam Google Sites (Insert → Embed → URL GitHub Pages).

## Mengapa lebih tepat daripada model Teachable Machine asal

| | Model TM asal | Model baharu |
|---|---|---|
| Backbone | MobileNet (kecil) | EfficientNetV2-B0 pratlatih ImageNet |
| Augmentasi latihan | tiada | potongan, putaran, cahaya, kontras, kabur (15× setiap imej) |
| Kelas tidak seimbang (9 imej "tak berdaftar") | tidak diambil kira | pemberat kelas + label smoothing |
| Objek bukan racun | dipaksa menjadi salah satu kelas | ditolak sebagai **TIDAK DIKENALI** (keserupaan prototaip) |
| Kamera langsung | satu bingkai | purata bergerak beberapa bingkai (lebih stabil) |

Penanda aras: 149 imej latihan yang dirosakkan (gelap/terang, senget, kabur, dizum) dan 100 imej bunga sebagai imej bukan racun.

| | Model TM asal | Model baharu |
|---|---|---|
| Ramalan betul (imej racun yang dirosakkan) | 91/149 (61%) | **131/149 (88%)**, 0 salah (18 ditanda "Tidak Pasti") |
| Imej bukan racun disangka racun | **95/100** | **0/100** |
| Imej ujian luar dalam repo (`antracol70wp/`, `basta15/`) | 6/6 | 6/6 |
| Validasi silang 5-lipatan | – | 99% |

## Melatih semula (tambah produk / tambah gambar)

1. Tambah kelas atau gambar dalam Teachable Machine, kemudian **Save project to file** (`.tm`).
2. Simpan fail itu di `training/data/hiper_app.tm`.
3. Kemas kini maklumat paparan produk dalam `INFO` di `training/train.py`.
4. Jalankan:

```bash
pip install tensorflow tensorflowjs scikit-learn pillow
python training/train.py training/data/hiper_app.tm --cv --extra antracol70wp="Antracol 70 WP" basta15="Basta 15"
python training/export_tfjs.py
```

Model TF.js baharu akan ditulis ke `docs/model/`.

> Petua: semakin banyak gambar sebenar (pelbagai sudut, cahaya, latar belakang) bagi setiap produk, terutamanya
> kelas **Racun tak berdaftar** (sekarang hanya 9 gambar), semakin tepat aplikasi.

## Penafian

Aplikasi ini ialah alat bantuan. Pengesahan rasmi status pendaftaran racun perlu dibuat berdasarkan nombor
pendaftaran pada label dan rekod Lembaga Racun Makhluk Perosak (LRMP).
