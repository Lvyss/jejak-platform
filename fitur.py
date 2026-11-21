"""Inti proyek: pra-proses, ekstraksi fitur (57 angka/foto), dan evaluasi."""
import glob, hashlib, os, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import kurtosis

UKURAN = 512
LABELS = ["O", "WA", "TG", "IG", "X"]
EKSTENSI = (".jpg", ".jpeg", ".png", ".webp", ".heic")


# ---------- pra-proses ----------
def crop_gray(im, size=UKURAN):
    """Crop tengah (offset kelipatan 8, tanpa resize) + grayscale."""
    im = im.convert("L")
    w, h = im.size
    if min(w, h) < size:
        raise ValueError(f"Foto {w}x{h} lebih kecil dari {size}px")
    left = ((w - size) // 2) // 8 * 8
    top = ((h - size) // 2) // 8 * 8
    im = im.crop((left, top, left + size, top + size))
    return np.asarray(im, dtype=np.float64)


def load(path):
    return crop_gray(Image.open(path))


# ---------- fitur ----------
def spektrum(img):
    w = np.outer(np.hanning(UKURAN), np.hanning(UKURAN))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * w))
    return np.log1p(np.abs(F))


def fitur_dft(img, P=None):
    P = spektrum(img) if P is None else P
    c = UKURAN // 2
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - c, xx - c).astype(int)
    bins = np.linspace(0, c, 33).astype(int)
    radial = [P[(r >= bins[i]) & (r < bins[i + 1])].mean() for i in range(32)]
    d = UKURAN // 8  # posisi puncak blok 8x8 (=64)
    puncak = np.mean([P[c, c + d], P[c, c - d], P[c + d, c], P[c - d, c]])
    sekitar = np.mean([P[c, c + d - 4], P[c, c - d + 4], P[c + d - 4, c], P[c - d + 4, c]])
    return radial + [puncak - sekitar]


def fitur_blockiness(img):
    dx = np.abs(np.diff(img, axis=1))
    dy = np.abs(np.diff(img, axis=0))
    return [dx[:, 7::8].mean() / (dx.mean() + 1e-9), dy[7::8, :].mean() / (dy.mean() + 1e-9)]


def fitur_noise(img):
    res = img - gaussian_filter(img, 1.0)
    return [res.std(), kurtosis(res.ravel())]


def fitur_tepi(img, ambang=50):
    mag = np.hypot(sobel(img, 0), sobel(img, 1))
    return [mag.mean(), (mag > ambang).mean()]


def fitur_hist(img):
    h, _ = np.histogram(img, bins=16, range=(0, 255), density=True)
    return list(h) + [img.mean(), img.std()]


KOLOM = ([f"dft_{i}" for i in range(33)]
         + ["blk_x", "blk_y", "noise_std", "noise_kurt", "edge_mean", "edge_frac"]
         + [f"hist_{i}" for i in range(16)] + ["lum_mean", "lum_std"])


def ekstrak(img):
    """Satu foto (array 512x512) -> list 57 angka + spektrum 2D."""
    P = spektrum(img)
    f = fitur_dft(img, P) + fitur_blockiness(img) + fitur_noise(img) + fitur_tepi(img) + fitur_hist(img)
    return f, P


# ---------- dataset ----------
def daftar_foto(folder):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "*", "*.*"))):
        if p.lower().endswith(EKSTENSI):
            out.append((os.path.basename(os.path.dirname(p)),
                        os.path.splitext(os.path.basename(p))[0].upper(), p))
    return out


def info_dataset(folder):
    """Format, ukuran, dan hash tiap file (cek awal sebelum ekstraksi)."""
    rows = []
    o_hash = {}
    for obj, lab, p in daftar_foto(folder):
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        if lab == "O":
            o_hash[obj] = h
    for obj, lab, p in daftar_foto(folder):
        im = Image.open(p)
        h = hashlib.md5(open(p, "rb").read()).hexdigest()
        rows.append(dict(obj=obj, label=lab, format=im.format, lebar=im.size[0], tinggi=im.size[1],
                         kb=round(os.path.getsize(p) / 1024, 1),
                         identik_O=(lab != "O" and o_hash.get(obj) == h)))
    return pd.DataFrame(rows)


def bangun_fitur(folder, progress=None):
    items = daftar_foto(folder)
    rows, spek, gagal = [], {}, []
    for i, (obj, lab, p) in enumerate(items):
        try:
            f, P = ekstrak(load(p))
        except Exception as e:  # foto terlalu kecil / rusak
            gagal.append((p, str(e)))
            continue
        rows.append([obj, lab] + f)
        s = spek.setdefault(lab, [np.zeros_like(P), 0])
        s[0] += P
        s[1] += 1
        if progress:
            progress((i + 1) / len(items))
    df = pd.DataFrame(rows, columns=["obj", "label"] + KOLOM)
    spek = {k: v[0] / v[1] for k, v in spek.items()}
    return df, spek, gagal


def buat_demo(folder, n_obj=12, seed=0):
    """Data sintetis agar aplikasi bisa dicoba sebelum foto asli siap.
    BUKAN pengganti data nyata."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    resep = {"O": (None, 95), "WA": (1000, 70), "TG": (1280, 85), "IG": (900, 55), "X": (1100, 40)}
    for k in range(n_obj):
        base = gaussian_filter(rng.normal(size=(1400, 1400)), rng.uniform(2, 12))
        base = (base - base.min()) / (np.ptp(base) + 1e-9) * 200 + 20
        base += rng.normal(0, rng.uniform(1, 5), base.shape)
        im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
        d = os.path.join(folder, f"obj{k+1:02d}")
        os.makedirs(d, exist_ok=True)
        for lab, (sisi, q) in resep.items():
            x = im if sisi is None else im.resize((sisi, sisi), Image.LANCZOS)
            x.save(os.path.join(d, f"{lab}.jpg"), quality=q)


# ---------- evaluasi ----------
def kelompok_fitur(cols):
    return {
        "DFT": [c for c in cols if c.startswith("dft_")],
        "Blockiness": ["blk_x", "blk_y"],
        "Noise": ["noise_std", "noise_kurt"],
        "Tepi": ["edge_mean", "edge_frac"],
        "Histogram": [c for c in cols if c.startswith(("hist", "lum"))],
        "Spasial (semua non-DFT)": [c for c in cols if c.startswith(("blk", "noise", "edge", "hist", "lum"))],
        "Gabungan": list(cols),
    }


def buat_model(nama, C=1.0, proba=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = LogisticRegression(C=C, max_iter=5000) if nama == "LogReg" else SVC(kernel="rbf", C=C, probability=proba)
    return make_pipeline(StandardScaler(), clf)


def evaluasi(df, C_lr=1.0, C_svm=1.0, n_splits=5):
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
    fitur = [c for c in df.columns if c not in ("obj", "label")]
    X, y, g = df[fitur].values, np.array(df["label"].tolist()), np.array(df["obj"].tolist())
    cv = GroupKFold(n_splits=min(n_splits, df["obj"].nunique()))
    labs = [l for l in LABELS if l in set(y)] + sorted(set(y) - set(LABELS))
    hasil = {"labels": labs, "model": {}, "fitur": {}}
    for nama, C in [("LogReg", C_lr), ("SVM", C_svm)]:
        m = buat_model(nama, C)
        pred = cross_val_predict(m, X, y, groups=g, cv=cv)
        skor = cross_val_score(m, X, y, groups=g, cv=cv)
        hasil["model"][nama] = dict(acc=accuracy_score(y, pred), fold=skor,
                                    cm=confusion_matrix(y, pred, labels=labs))
    lr = buat_model("LogReg", C_lr)
    for nama, cols in kelompok_fitur(fitur).items():
        pred = cross_val_predict(lr, df[cols].values, y, groups=g, cv=cv)
        hasil["fitur"][nama] = (accuracy_score(y, pred), len(cols))
    lr.fit(X, y)
    coef = pd.DataFrame(lr[-1].coef_, index=lr[-1].classes_, columns=fitur)
    hasil["koef"] = coef
    hasil["acak"] = 1 / len(labs)
    return hasil
