# Artefak publik SwinIR

- [Dashboard interaktif](dashboard/index.html): hasil agregat, sweep, log numerik, konfigurasi, dan statistik. Buka berkas HTML setelah mengunduh repo; tidak membutuhkan layanan eksternal.
- [Paper bahasa Indonesia](paper/paper_ieee_id.html): adaptasi HTML dua kolom IEEE, 7 halaman termasuk lampiran, dengan 12 referensi. Kolom penulis sengaja belum diisi.
- [Tabel CSV](results/): hasil utama, seluruh sweep, dan seluruh perbandingan statistik.

Sumber angka: [notebook Kaggle versi 5](https://www.kaggle.com/code/kanzaaz/swinlr-sesi2), run 357177881. Konfirmasi memakai seed 1–3, 1.758 citra test, dan 10.000 iterasi training. Sobel statis λ=1 menghasilkan kualitas tertinggi pada kondisi konfirmasi; adaptif λ=2 mengungguli L1 tetapi tidak mengungguli statis.

Gambar sumber/target/prediksi, contoh citra yang tertanam pada paper lokal, metrik per citra, arsip Kaggle, checkpoint, serta metadata lingkungan lengkap tidak didistribusikan. Lisensi dan provenance dataset belum terverifikasi. Grafik yang disertakan hanya visualisasi angka hasil eksperimen. Versi lengkap tetap berada di folder lokal yang diabaikan Git.

Untuk membangun ulang ekspor numerik setelah menyediakan arsip hasil lokal pada `runs/kaggle-swinlr-sesi2-v5/`, jalankan `python scripts/prepare_public_artifacts.py`. Sumber paper lokal opsional pada `docs/paper/paper_ieee_id.html`; proses menyingkirkan satu gambar contoh yang dikenali, dan berhenti jika struktur gambar berubah. Tinjau ulang ekspor sebelum publikasi.

Nilai agregat identik dengan hasil eksperimen; pengecualian gambar tidak mengubah angka penelitian. Format HTML adalah draf peninjauan, bukan template resmi untuk pengiriman IEEE.
