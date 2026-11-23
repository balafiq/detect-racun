"""
Latih semula pengelas racun berdaftar / tidak berdaftar.

Input : fail projek Teachable Machine (.tm, iaitu zip berisi imej "<Kelas>-!-<n>.jpg")
Output: model TF.js (web/model/) + metadata.json

Pendekatan (lebih kukuh berbanding Teachable Machine asal):
  * Backbone pratlatih yang lebih besar (EfficientNetV2-B0 ImageNet) dibekukan,
  * Banyak augmentasi (potong rawak, putaran, skala, warna, kabur) supaya model
    tahan perubahan cahaya / sudut kamera,
  * Kepala linear dengan label smoothing + class weight (kelas "tak berdaftar" sedikit imej),
  * Prototaip (purata embedding) setiap kelas disimpan untuk pengesanan "tidak dikenali"
    (open-set) - imej yang tidak menyerupai mana-mana racun berdaftar ditolak.
"""
import argparse, glob, io, json, os, random, re, zipfile
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
import keras
from sklearn.model_selection import StratifiedKFold

IMG = 224
UNREG = "Racun tak berdaftar"

# Maklumat paparan setiap kelas
INFO = {
    "Antracol 70 WP": {"nama": "Antracol 70 WP", "jenis": "Racun kulat (Propineb)", "berdaftar": True},
    "Basta 15": {"nama": "Basta 15", "jenis": "Racun rumpai (Glufosinate-ammonium)", "berdaftar": True},
    "Dithane Gr": {"nama": "Dithane GR", "jenis": "Racun kulat (Mancozeb)", "berdaftar": True},
    "Kencis": {"nama": "Kencis", "jenis": "Racun serangga (Cypermethrin)", "berdaftar": True},
    "amistar": {"nama": "Amistar", "jenis": "Racun kulat (Azoxystrobin)", "berdaftar": True},
    "ecomax": {"nama": "Ecomax", "jenis": "Racun makhluk perosak", "berdaftar": True},
    UNREG: {"nama": "Racun tidak berdaftar", "jenis": "Tiada label / tiada no. pendaftaran", "berdaftar": False},
}


def load_tm(path):
    z = zipfile.ZipFile(path)
    data = {}
    for n in z.namelist():
        m = re.match(r"(.+)-!-(\d+)\.(jpg|jpeg|png)$", n)
        if not m:
            continue
        im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        data.setdefault(m.group(1), []).append(im)
    return data


def augment(im, rng):
    w, h = im.size
    # potong rawak (zoom 60-100%)
    s = rng.uniform(0.6, 1.0)
    cw, ch = int(w * s), int(h * s * rng.uniform(0.85, 1.15))
    ch = min(ch, h)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    im = im.crop((x, y, x + cw, y + ch))
    im = im.rotate(rng.uniform(-20, 20), resample=Image.BILINEAR, expand=False, fillcolor=tuple(rng.randint(0, 255) for _ in range(3)))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.7, 1.3))
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    return im


def center(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def to_arr(ims):
    return np.stack([np.asarray(center(i).resize((IMG, IMG), Image.BILINEAR), dtype=np.float32) for i in ims])


def build_backbone():
    # Input: piksel 0..255 RGB. Pra-pemprosesan berada dalam model (EfficientNetV2 include_preprocessing).
    base = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(IMG, IMG, 3), pooling="avg", include_preprocessing=True)
    base.trainable = False
    return base


def embed(backbone, ims, bs=32):
    return backbone.predict(to_arr(ims), batch_size=bs, verbose=0)


def make_head(dim, ncls):
    inp = keras.Input((dim,))
    x = keras.layers.Dropout(0.3)(inp)
    out = keras.layers.Dense(ncls, activation="softmax", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    m = keras.Model(inp, out)
    m.compile(keras.optimizers.Adam(1e-3), keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    return m


def fit_head(X, y, ncls, epochs=60):
    counts = np.bincount(y, minlength=ncls)
    cw = {i: len(y) / (ncls * max(c, 1)) for i, c in enumerate(counts)}
    m = make_head(X.shape[1], ncls)
    m.fit(X, keras.utils.to_categorical(y, ncls), epochs=epochs, batch_size=64, class_weight=cw, verbose=0, shuffle=True)
    return m


def aug_features(backbone, ims, labels, n_aug, seed):
    rng = random.Random(seed)
    all_ims, all_y = [], []
    for im, l in zip(ims, labels):
        all_ims.append(im); all_y.append(l)
        for _ in range(n_aug):
            all_ims.append(augment(im, rng)); all_y.append(l)
    return embed(backbone, all_ims), np.array(all_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tm")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "model"))
    ap.add_argument("--extra", nargs="*", default=[], help="folder ujian luar: folder=kelas")
    ap.add_argument("--aug", type=int, default=15)
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()

    data = load_tm(a.tm)
    classes = sorted(data, key=lambda c: (c == UNREG, c.lower()))
    print({c: len(v) for c, v in data.items()})
    ims = [im for c in classes for im in data[c]]
    y = np.array([i for i, c in enumerate(classes) for _ in data[c]])
    backbone = build_backbone()

    if a.cv:
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        accs = []
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            Xtr, ytr = aug_features(backbone, [ims[i] for i in tr], y[tr], a.aug, k)
            head = fit_head(Xtr, ytr, len(classes))
            # uji pada imej yang telah diubah (simulasi kamera)
            rng = random.Random(100 + k)
            te_ims = [augment(ims[i], rng) for i in te]
            p = head.predict(embed(backbone, te_ims), verbose=0).argmax(1)
            p0 = head.predict(embed(backbone, [ims[i] for i in te]), verbose=0).argmax(1)
            accs.append(((p0 == y[te]).mean(), (p == y[te]).mean()))
            print(f"fold {k}: asal={accs[-1][0]:.3f} diubah={accs[-1][1]:.3f}")
        print("purata CV (asal, diubah):", np.mean(accs, 0))

    X, Y = aug_features(backbone, ims, y, a.aug, 42)
    head = fit_head(X, Y, len(classes))

    # Prototaip untuk pengesanan open-set (cosine similarity)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    protos = np.stack([Xn[Y == i].mean(0) for i in range(len(classes))])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    sims = (Xn @ protos.T)[np.arange(len(Y)), Y]
    print("keserupaan latihan persentil 2:", float(np.percentile(sims, 2)))
    # Dikalibrasi dengan 300 imej bukan-racun (bunga): maks 0.34. Imej racun sebenar biasanya > 0.45.
    sim_thr = 0.40

    # Model penuh: piksel 0..255 -> [embedding, kebarangkalian]
    inp = keras.Input((IMG, IMG, 3), name="image")
    emb = backbone(inp)
    prob = head(emb)
    norm = keras.layers.UnitNormalization(name="emb_norm")(emb)
    sim = keras.layers.Dense(len(classes), use_bias=False, name="proto_sim")(norm)
    full = keras.Model(inp, [prob, sim])
    full.get_layer("proto_sim").set_weights([protos.T.astype(np.float32)])

    for spec in a.extra:
        folder, cls = spec.split("=")
        fs = [f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        ext = to_arr([Image.open(f).convert("RGB") for f in fs])
        pr, sm = full.predict(ext, verbose=0)
        for f, p, s in zip(fs, pr, sm):
            print(f"[ujian luar] {os.path.basename(f)} jangka={classes[p.argmax()]} ({p.max():.2f}) sim={s.max():.2f} betul={classes[p.argmax()] == cls}")

    os.makedirs(a.out, exist_ok=True)
    full.save(os.path.join(a.out, "..", "..", "training", "model.keras"))
    meta = {
        "labels": classes,
        "info": [INFO.get(c, {"nama": c, "jenis": "", "berdaftar": c != UNREG}) for c in classes],
        "imageSize": IMG,
        "simThreshold": sim_thr,
        "probThreshold": 0.55,
    }
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=2, ensure_ascii=False)
    print("siap")


if __name__ == "__main__":
    main()
