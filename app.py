import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import fitur as F

st.set_page_config(page_title="Jejak Platform", page_icon="🔬", layout="wide")

BG, PANEL, INK, MUTED, ACC, AMB = "#0F1620", "#17212E", "#E6EAF2", "#8FA0B8", "#8B9CFF", "#FFB454"
WARNA = {"O": "#E6EAF2", "WA": "#5EE0A0", "TG": "#5CC8FF", "IG": "#FF7AB6", "X": "#FFB454"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3 {{ font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hero {{ padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;
  background: radial-gradient(120% 140% at 0% 0%, #243352 0%, {PANEL} 55%, {BG} 100%);
  border: 1px solid #26354d; }}
.hero h1 {{ font-size: 2.1rem; margin: 0 0 6px 0; }}
.hero p {{ color: {MUTED}; margin: 0; max-width: 62ch; line-height: 1.6; }}
.step {{ background: {PANEL}; border: 1px solid #26354d; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.step b {{ font-family: 'Sora'; }}
.step small {{ color: {MUTED}; }}
.ok {{ color: #5EE0A0; }} .wait {{ color: {AMB}; }}
div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #26354d; padding: 14px 16px; border-radius: 14px; }}
.stButton > button {{ border-radius: 10px; font-weight: 600; }}
.note {{ background: #1d2a3d; border-left: 3px solid {AMB}; padding: 10px 14px; border-radius: 8px; color: {INK}; font-size: .92rem; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid #26354d; }}
</style>
""", unsafe_allow_html=True)


def gaya(ax):
    ax.set_facecolor("none")
    for s in ax.spines.values():
        s.set_color("#33445f")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED); ax.yaxis.label.set_color(MUTED); ax.title.set_color(INK)


def fig(w=6, h=3.2, **kw):
    f, ax = plt.subplots(figsize=(w, h), **kw)
    f.patch.set_alpha(0)
    return f, ax


# ---------- state ----------
st.session_state.setdefault("folder", "data")
os.makedirs("hasil", exist_ok=True)
if "df" not in st.session_state and os.path.exists("fitur.csv"):
    st.session_state.df = pd.read_csv("fitur.csv")
    if os.path.exists("hasil/spektrum.npz"):
        z = np.load("hasil/spektrum.npz")
        st.session_state.spek = {k: z[k] for k in z.files}

df = st.session_state.get("df")
nfoto = len(F.daftar_foto(st.session_state.folder)) if os.path.isdir(st.session_state.folder) else 0

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 🔬 Jejak Platform")
    st.caption("Deteksi asal platform foto lewat DFT 2D + fitur spasial")
    hal = st.radio("Langkah", ["Beranda", "1 · Dataset", "2 · Ekstraksi fitur", "3 · Evaluasi", "4 · Uji foto"],
                   label_visibility="collapsed")
    st.divider()
    st.session_state.folder = st.text_input("Folder data", st.session_state.folder,
                                            help="Isinya: obj01/O.jpg, WA.jpg, TG.jpg, IG.jpg, X.jpg, dst.")
    st.caption(f"{nfoto} foto terdeteksi" if nfoto else "Folder belum ada / kosong")

# ---------- Beranda ----------
if hal == "Beranda":
    st.markdown("""<div class="hero"><h1>Foto ini pernah lewat platform mana?</h1>
    <p>Kirim satu foto lewat WA, Telegram, Instagram, dan X. Tiap platform mengompres dengan caranya sendiri
    dan meninggalkan jejak di spektrum frekuensi dan detail halus. Aplikasi ini mengukur jejak itu,
    melatih model, dan menunjukkan seberapa jauh jejaknya bisa dibedakan.</p></div>""", unsafe_allow_html=True)
    langkah = [("1 · Dataset", nfoto > 0, f"{nfoto} foto" if nfoto else "Siapkan folder foto atau buat data demo"),
               ("2 · Ekstraksi", df is not None, f"{len(df)} baris fitur" if df is not None else "Ubah foto jadi 57 angka"),
               ("3 · Evaluasi", "hasil" in st.session_state, "Akurasi, confusion matrix, fitur terbaik"),
               ("4 · Uji foto", "hasil" in st.session_state, "Tebak platform foto baru")]
    cols = st.columns(4)
    for c, (judul, sudah, ket) in zip(cols, langkah):
        c.markdown(f'<div class="step"><b>{judul}</b><br><span class="{"ok" if sudah else "wait"}">'
                   f'{"Siap" if sudah else "Belum"}</span><br><small>{ket}</small></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="note">Mulai dari <b>1 · Dataset</b>. Belum punya foto? Di sana ada tombol data demo '
                'supaya kamu bisa mencoba seluruh alur dulu. Data demo sintetis, bukan untuk laporan.</div>',
                unsafe_allow_html=True)

# ---------- 1 Dataset ----------
elif hal == "1 · Dataset":
    st.header("Dataset")
    st.caption("Cek format, ukuran, dan apakah ada platform yang tidak mengubah foto sama sekali.")
    if st.button("Buat data demo (12 objek)"):
        F.buat_demo(st.session_state.folder)
        st.rerun()
    if nfoto == 0:
        st.info(f"Belum ada foto di folder “{st.session_state.folder}”. Susun seperti ini: obj01/O.jpg, WA.jpg, TG.jpg, IG.jpg, X.jpg.")
        st.stop()
    info = F.info_dataset(st.session_state.folder)
    c = st.columns(4)
    c[0].metric("Foto", len(info)); c[1].metric("Objek", info.obj.nunique())
    c[2].metric("Platform", info.label.nunique()); c[3].metric("Identik dengan O", int(info.identik_O.sum()))
    ringkas = info.groupby("label").agg(jumlah=("obj", "count"), format=("format", lambda s: ", ".join(sorted(set(s)))),
                                        lebar_rata2=("lebar", "mean"), kb_rata2=("kb", "mean")).round(0)
    ringkas = ringkas.reindex([l for l in F.LABELS if l in ringkas.index])
    st.subheader("Ringkasan per platform")
    st.dataframe(ringkas, width="stretch")
    o_w = info[info.label == "O"].lebar.median() if (info.label == "O").any() else None
    notes = []
    for lab, r in ringkas.iterrows():
        if o_w and lab != "O" and abs(r.lebar_rata2 - o_w) > 1:
            notes.append(f"{lab} mengubah ukuran foto (≈{int(r.lebar_rata2)}px vs {int(o_w)}px di O): ada resize, grid blok bergeser.")
        if lab != "O" and info[info.label == lab].identik_O.any():
            notes.append(f"{lab}: sebagian file identik dengan O (tidak diubah platform). Itu temuan sah.")
        if "JPEG" not in r["format"] and lab != "O":
            notes.append(f"{lab} bukan JPEG ({r['format']}): fitur blockiness 8×8 akan melemah.")
    kecil = info[(info.lebar < 512) | (info.tinggi < 512)]
    if len(kecil):
        notes.append(f"{len(kecil)} foto lebih kecil dari 512px dan akan dilewati.")
    for n in notes:
        st.markdown(f'<div class="note">{n}</div>', unsafe_allow_html=True); st.markdown("")
    with st.expander("Tabel lengkap"):
        st.dataframe(info, width="stretch")

# ---------- 2 Ekstraksi ----------
elif hal == "2 · Ekstraksi fitur":
    st.header("Ekstraksi fitur")
    st.caption("Tiap foto: crop tengah 512×512, grayscale, lalu 57 angka (DFT 33, blockiness 2, noise 2, tepi 2, histogram 18).")
    if nfoto == 0:
        st.info("Siapkan dataset dulu di langkah 1."); st.stop()
    if st.button("Jalankan ekstraksi", type="primary"):
        bar = st.progress(0.0, text="Menghitung fitur…")
        d, spek, gagal = F.bangun_fitur(st.session_state.folder, lambda p: bar.progress(p, text="Menghitung fitur…"))
        d.to_csv("fitur.csv", index=False); np.savez("hasil/spektrum.npz", **spek)
        st.session_state.df, st.session_state.spek = d, spek
        st.session_state.pop("hasil", None)
        bar.empty()
        for p, e in gagal:
            st.warning(f"Dilewati: {p} ({e})")
        st.rerun()
    if df is None:
        st.stop()
    st.success(f"fitur.csv siap: {len(df)} baris × {df.shape[1]} kolom")
    st.download_button("Unduh fitur.csv", df.to_csv(index=False), "fitur.csv", "text/csv")
    spek = st.session_state.get("spek", {})
    labs = [l for l in F.LABELS if l in spek]
    if labs:
        st.subheader("Spektrum 2D rata-rata per platform")
        cols = st.columns(len(labs))
        for c, l in zip(cols, labs):
            f, ax = fig(2.4, 2.4); ax.imshow(spek[l], cmap="magma", vmin=np.percentile(spek[l], 2)); ax.axis("off")
            c.pyplot(f, width="stretch"); c.markdown(f"<center><b>{l}</b></center>", unsafe_allow_html=True)
        st.caption("Titik terang di tengah = frekuensi rendah. Puncak kecil di jarak 64 piksel dari tengah = jejak blok JPEG 8×8.")
        st.subheader("Kurva radial rata-rata")
        f, ax = fig(8, 3.2)
        for l in labs:
            sub = df[df.label == l][[f"dft_{i}" for i in range(32)]].mean()
            ax.plot(range(32), sub.values, label=l, color=WARNA.get(l), lw=2)
        ax.set_xlabel("tingkat kerapatan pola (0 = kasar, 31 = halus)"); ax.set_ylabel("energi (log)")
        ax.legend(frameon=False, labelcolor=INK, ncol=5); gaya(ax)
        st.pyplot(f, width="stretch")
    st.subheader("Rata-rata fitur tunggal per platform")
    st.dataframe(df.groupby("label")[["blk_x", "blk_y", "noise_std", "edge_mean", "edge_frac", "lum_mean"]].mean()
                 .reindex(labs or None).round(3), width="stretch")

# ---------- 3 Evaluasi ----------
elif hal == "3 · Evaluasi":
    st.header("Evaluasi")
    st.caption("Uji jujur: GroupKFold per objek, jadi model selalu diuji pada objek yang belum pernah dilihatnya.")
    if df is None:
        st.info("Jalankan ekstraksi dulu di langkah 2."); st.stop()
    if df.obj.nunique() < 3:
        st.warning("Butuh minimal 3 objek."); st.stop()
    c1, c2, c3 = st.columns(3)
    C_lr = c1.select_slider("C LogReg", [0.01, 0.1, 1.0, 10.0], 1.0)
    C_svm = c2.select_slider("C SVM", [0.01, 0.1, 1.0, 10.0], 1.0)
    k = c3.slider("Jumlah fold", 3, 10, 5)
    if st.button("Jalankan evaluasi", type="primary"):
        with st.spinner("Melatih dan menguji…"):
            st.session_state.hasil = F.evaluasi(df, C_lr, C_svm, k)
            st.session_state.C = (C_lr, C_svm)
    h = st.session_state.get("hasil")
    if not h:
        st.stop()
    m = st.columns(3)
    for c, n in zip(m[:2], ["LogReg", "SVM"]):
        r = h["model"][n]; c.metric(n, f"{r['acc']*100:.1f}%", f"± {r['fold'].std()*100:.1f} antar fold", delta_color="off")
    m[2].metric("Tebak acak", f"{h['acak']*100:.0f}%")
    st.markdown('<div class="note">Mirip → polanya sederhana. SVM jauh lebih tinggi → ada pola tak lurus. '
                'Keduanya rendah → masalah di fitur atau data, bukan model.</div>', unsafe_allow_html=True)
    st.subheader("Confusion matrix")
    cols = st.columns(2)
    for c, n in zip(cols, ["LogReg", "SVM"]):
        cm = h["model"][n]["cm"]; f, ax = fig(4.2, 3.6)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(h["labels"])), h["labels"]); ax.set_yticks(range(len(h["labels"])), h["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#0F1620", fontsize=10)
        ax.set_xlabel("tebakan"); ax.set_ylabel("sebenarnya"); ax.set_title(n); gaya(ax)
        c.pyplot(f, width="stretch")
    st.subheader("Fitur mana yang paling membedakan?")
    ft = pd.DataFrame({"akurasi": {k: v[0] for k, v in h["fitur"].items()},
                       "jumlah kolom": {k: v[1] for k, v in h["fitur"].items()}}).sort_values("akurasi")
    f, ax = fig(8, 3.6)
    ax.barh(ft.index, ft.akurasi * 100, color=[AMB if i == ft.akurasi.idxmax() else ACC for i in ft.index])
    ax.axvline(h["acak"] * 100, color=MUTED, ls="--", lw=1); ax.set_xlabel("akurasi LogReg (%)"); gaya(ax)
    for i, (n, r) in enumerate(ft.iterrows()):
        ax.text(r.akurasi * 100 + 1, i, f"{r.akurasi*100:.0f}%  ({int(r['jumlah kolom'])} kolom)", va="center", color=INK, fontsize=8)
    ax.set_xlim(0, 118); st.pyplot(f, width="stretch")
    st.caption("Garis putus = tebak acak. DFT punya 33 kolom vs blockiness 2, jadi sebagian keunggulan DFT bisa karena kolomnya lebih banyak. Catat di laporan.")
    st.subheader("Fitur individual paling berpengaruh (bobot LogReg)")
    imp = h["koef"].abs().mean().sort_values(ascending=False).head(10)[::-1]
    f, ax = fig(8, 3.4); ax.barh(imp.index, imp.values, color=ACC); ax.set_xlabel("rata-rata |bobot| antar kelas"); gaya(ax)
    st.pyplot(f, width="stretch")
    st.caption("Kolom DFT saling berkorelasi sehingga bobotnya terbagi-bagi. Anggap petunjuk, bukan bukti pasti.")
    with st.expander("Akurasi per fold"):
        st.write({n: np.round(h["model"][n]["fold"], 3).tolist() for n in h["model"]})

# ---------- 4 Uji foto ----------
else:
    st.header("Uji foto")
    st.caption("Unggah foto (minimal 512px) dan model menebak platformnya. Hasil hanya sah untuk kondisi data latihmu.")
    if df is None:
        st.info("Jalankan ekstraksi dulu di langkah 2."); st.stop()
    up = st.file_uploader("Foto", type=["jpg", "jpeg", "png", "webp"])
    nama = st.radio("Model", ["LogReg", "SVM"], horizontal=True)
    if up:
        try:
            im = Image.open(up); img = F.crop_gray(im)
        except Exception as e:
            st.error(f"Foto tidak bisa diproses: {e}"); st.stop()
        fitur_, P = F.ekstrak(img)
        fcols = [c for c in df.columns if c not in ("obj", "label")]
        mdl = F.buat_model(nama, (st.session_state.get("C") or (1.0, 1.0))[0 if nama == "LogReg" else 1], proba=True)
        mdl.fit(df[fcols].values, np.array(df["label"].tolist()))
        pr = pd.Series(mdl.predict_proba(pd.DataFrame([fitur_], columns=fcols).values)[0], index=mdl.classes_)
        pr = pr.reindex([l for l in F.LABELS if l in pr.index])
        a, b = st.columns([1, 1])
        a.image(im, caption=f"{im.format} · {im.size[0]}×{im.size[1]}", width="stretch")
        top = pr.idxmax()
        b.metric("Tebakan", top, f"{pr.max()*100:.0f}% yakin", delta_color="off")
        f, ax = fig(4.5, 2.6); ax.bar(pr.index, pr.values * 100, color=[WARNA.get(i, ACC) for i in pr.index])
        ax.set_ylabel("peluang (%)"); gaya(ax); b.pyplot(f, width="stretch")
        f, ax = fig(3, 3); ax.imshow(P, cmap="magma", vmin=np.percentile(P, 2)); ax.axis("off"); ax.set_title("Spektrum foto ini")
        gaya(ax); st.pyplot(f)
        if pr.max() < 0.5:
            st.markdown('<div class="note">Peluang tertinggi di bawah 50%: model tidak yakin. Foto ini mungkin lewat jalur '
                        'yang tidak ada di data latihmu.</div>', unsafe_allow_html=True)
