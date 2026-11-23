/* HIPER Apps – Pengesan Racun Berdaftar
 * Model: EfficientNetV2-B0 (pratlatih) + pengelas yang dilatih semula daripada data Teachable Machine.
 * Model mengeluarkan dua tensor: kebarangkalian kelas & keserupaan kosinus dengan prototaip setiap kelas.
 * Keserupaan digunakan untuk menolak objek yang tidak dikenali (supaya tidak "dipaksa" jadi salah satu kelas).
 */
(() => {
  const $ = (id) => document.getElementById(id);
  const MODEL_URL = "model/model.json";
  const META_URL = "model/metadata.json";
  const SMOOTH = 0.35; // pemberat EMA untuk kamera (lebih kecil = lebih stabil)

  let model, meta, probIdx = 0, ready = false, nResults = 0;
  let stream = null, facing = "environment", running = false, ema = null, busy = false;
  const work = $("work");
  const ctx = work.getContext("2d", { willReadFrequently: true });

  // ---------- Model ----------
  function loading(text) {
    $("btnStart").disabled = true;
    $("btnStart").textContent = "⏳ " + text;
    $("statusSub").textContent = text;
  }

  /** Input ujian deterministik (sama seperti dalam training/train.py) */
  function selfTestInput() {
    return tf.tidy(() => {
      const n = meta.imageSize;
      const i = tf.range(0, n).reshape([n, 1, 1]).mul(7);
      const j = tf.range(0, n).reshape([1, n, 1]).mul(13);
      const c = tf.tensor1d([0, 50, 100]).reshape([1, 1, 3]);
      return tf.mod(i.add(j).add(c), 256).toFloat().expandDims(0);
    });
  }

  /** Jalankan model pada input ujian; pastikan keputusan sama seperti semasa latihan */
  async function selfTest() {
    const x = selfTestInput();
    const outs = model.execute(x);
    x.dispose();
    const arrs = await Promise.all(outs.map((t) => t.data()));
    outs.forEach((t) => t.dispose());
    const ref = meta.selfTest.prob;
    const diff = (a) => Math.max(...Array.from(a).map((v, k) => Math.abs(v - ref[k])));
    const d0 = diff(arrs[0]), d1 = diff(arrs[1]);
    probIdx = d0 <= d1 ? 0 : 1;
    const d = Math.min(d0, d1);
    return Number.isFinite(d) && d < 0.05;
  }

  async function init() {
    try {
      loading("Memuatkan model AI…");
      if (tf.wasm && tf.wasm.setWasmPaths) tf.wasm.setWasmPaths(new URL("lib/", location.href).href);
      try { await tf.setBackend("webgl"); } catch (_) { /* guna lalai */ }
      await tf.ready();
      [model, meta] = await Promise.all([
        tf.loadGraphModel(MODEL_URL, { onProgress: (f) => loading("Memuat turun model… " + Math.round(f * 100) + "%") }),
        fetch(META_URL).then((r) => r.json()),
      ]);
      $("knownList").textContent = meta.info.filter((i) => i.berdaftar).map((i) => i.nama).join(", ");
      // Pilih backend yang memberi keputusan betul pada peranti ini (WebGL → WASM → CPU)
      let ok = false;
      for (const be of ["webgl", "wasm", "cpu"]) {
        try {
          if (tf.getBackend() !== be && !(await tf.setBackend(be))) continue;
          await tf.ready();
          loading("Menyediakan model (" + be + ")…");
          if (await selfTest()) { ok = true; break; }
          console.warn("Ujian kendiri gagal pada backend", be);
        } catch (e) { console.warn("Backend", be, "gagal:", e); }
      }
      if (!ok) throw new Error("Model tidak dapat dijalankan dengan betul pada peranti ini.");
      $("backend").textContent = tf.getBackend();
      ready = true;
      $("btnStart").disabled = false;
      $("btnStart").textContent = "▶ Mula Kamera";
      setIdle("Sedia", "Mulakan kamera atau muat naik gambar");
    } catch (e) {
      console.error(e);
      $("btnStart").textContent = "⚠️ Model gagal dimuatkan";
      setState("warn", "⚠️", "Gagal memuatkan model", String(e.message || e) + " – muat semula halaman.");
      $("camHelp").classList.remove("hidden");
      $("camHelpText").textContent = "Model AI gagal dimuatkan: " + (e.message || e) + ". Semak sambungan internet dan muat semula halaman.";
    }
  }

  /** Lukis kawasan segi empat dari sumber ke kanvas 224x224 */
  function drawCrop(src, sw, sh, zoom = 1) {
    const s = Math.min(sw, sh) * zoom;
    const sx = (sw - s) / 2, sy = (sh - s) / 2;
    ctx.drawImage(src, sx, sy, s, s, 0, 0, work.width, work.height);
    return tf.browser.fromPixels(work).toFloat();
  }

  /** Ramal daripada satu atau lebih imej 224x224 (purata) */
  async function predict(tensors) {
    const batch = tf.stack(tensors);
    tensors.forEach((t) => t.dispose());
    const outs = model.execute(batch);
    batch.dispose();
    const prob = await outs[probIdx].mean(0).data();
    const sim = await outs[1 - probIdx].max(0).data();
    outs.forEach((t) => t.dispose());
    return { prob: Array.from(prob), sim: Array.from(sim) };
  }

  // ---------- Keputusan ----------
  function decide({ prob, sim }) {
    let top = 0;
    prob.forEach((p, i) => { if (p > prob[top]) top = i; });
    const info = meta.info[top];
    const p = prob[top];
    const bestSim = Math.max(...sim);
    if (bestSim < meta.simThreshold) {
      return { kind: "warn", icon: "❓", label: "TIDAK DIKENALI", sub: "Tiada racun dikesan / produk tiada dalam pangkalan data. Halakan kamera ke label.", p: null };
    }
    if (!info.berdaftar && p >= 0.5) {
      return { kind: "bad", icon: "⛔", label: "TIDAK BERDAFTAR", sub: `${info.jenis}. Jangan guna – laporkan kepada pihak berkuasa.`, p };
    }
    if (info.berdaftar && p >= meta.probThreshold) {
      return { kind: "ok", icon: "✅", label: "BERDAFTAR", sub: `${info.nama}${info.jenis ? " · " + info.jenis : ""}`, p };
    }
    return { kind: "warn", icon: "⚠️", label: "TIDAK PASTI", sub: `Mungkin ${info.nama}. Dekatkan kamera ke label atau semak no. pendaftaran secara manual.`, p };
  }

  function setState(kind, icon, label, sub, p = null) {
    const r = $("result");
    r.className = "card result " + kind;
    $("statusIcon").textContent = icon;
    $("statusLabel").textContent = label;
    $("statusSub").textContent = sub;
    $("confWrap").classList.toggle("hidden", p == null);
    if (p != null) {
      $("confPct").textContent = Math.round(p * 100) + "%";
      $("confBar").style.width = (p * 100).toFixed(1) + "%";
    }
  }
  const setIdle = (label, sub) => setState("idle", "🔍", label, sub);

  function render(res) {
    nResults++;
    const d = decide(res);
    setState(d.kind, d.icon, d.label, d.sub, d.p);
    if (running) {
      const pill = $("pill");
      pill.className = "pill " + d.kind;
      pill.textContent = d.kind === "ok" ? `✅ ${meta.info[res.prob.indexOf(Math.max(...res.prob))].nama}` : `${d.icon} ${d.label}`;
    }
    const order = res.prob.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    $("probs").innerHTML = order.map(([p, i]) => {
      const inf = meta.info[i];
      return `<li><span>${inf.berdaftar ? "✅" : "⛔"} ${inf.nama}</span><div class="mini"><div style="width:${(p * 100).toFixed(1)}%"></div></div><b>${Math.round(p * 100)}%</b></li>`;
    }).join("") + `<li><span>Keserupaan maks.</span><div></div><b>${Math.max(...res.sim).toFixed(2)}</b></li>`;
  }

  // ---------- Kamera ----------
  const inIframe = (() => { try { return window.self !== window.top; } catch (_) { return true; } })();

  function camError(msg) {
    setState("warn", "⚠️", "Kamera tidak dapat dibuka", msg);
    $("camHelp").classList.remove("hidden");
    $("camHelpText").textContent = msg;
  }

  async function startCam() {
    $("camHelp").classList.add("hidden");
    if (!ready) {
      setState("idle", "⏳", "Model masih dimuatkan…", "Sila tunggu sehingga status 'Sedia', kemudian tekan semula.");
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      camError(inIframe
        ? "Kamera disekat kerana aplikasi dibuka dalam halaman lain (embed). Tekan 'Buka dalam tab baharu' atau guna 'Ambil Gambar'."
        : "Pelayar ini tidak menyokong kamera langsung (perlu HTTPS & pelayar terkini seperti Chrome/Safari). Guna 'Ambil Gambar'.");
      return;
    }
    $("btnStart").disabled = true;
    setState("idle", "⏳", "Membuka kamera…", "Benarkan akses kamera jika diminta.");
    const tries = [
      { video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false },
      { video: { facingMode: facing }, audio: false },
      { video: true, audio: false },
    ];
    let err = null;
    for (const c of tries) {
      try { stream = await navigator.mediaDevices.getUserMedia(c); err = null; break; }
      catch (e) { err = e; if (e && (e.name === "NotAllowedError" || e.name === "SecurityError")) break; }
    }
    $("btnStart").disabled = false;
    if (!stream) {
      const n = err && err.name;
      let msg;
      if (n === "NotAllowedError" || n === "SecurityError") {
        msg = inIframe
          ? "Akses kamera disekat oleh halaman embed (cth. Google Sites). Tekan 'Buka dalam tab baharu' atau guna 'Ambil Gambar'."
          : "Akses kamera ditolak. Tekan ikon 🔒 di sebelah alamat laman → Kebenaran → Kamera → Benarkan, kemudian muat semula halaman.";
      } else if (n === "NotFoundError" || n === "OverconstrainedError") {
        msg = "Tiada kamera ditemui pada peranti ini. Guna 'Ambil Gambar' atau 'Muat Naik Gambar'.";
      } else if (n === "NotReadableError" || n === "AbortError") {
        msg = "Kamera sedang digunakan oleh aplikasi lain. Tutup aplikasi lain (cth. WhatsApp/Zoom) dan cuba lagi.";
      } else {
        msg = "Ralat kamera" + (n ? " (" + n + ")" : "") + ". Guna 'Ambil Gambar' atau 'Muat Naik Gambar'.";
      }
      camError(msg);
      return;
    }
    const v = $("video");
    v.srcObject = stream;
    try { await v.play(); } catch (e) { /* sesetengah pelayar perlukan sentuhan; video tetap dimainkan selepas ini */ }
    const settings = stream.getVideoTracks()[0].getSettings();
    v.classList.toggle("mirror", settings.facingMode === "user");
    $("camPlaceholder").classList.add("hidden");
    $("btnStart").textContent = "■ Henti Kamera";
    $("btnStart").classList.add("stop");
    $("btnFlip").disabled = false;
    running = true; ema = null;
    setState("idle", "🔍", "Mengimbas…", "Letakkan label racun di dalam kotak.");
    loop();
  }

  function stopCam() {
    running = false;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    $("video").srcObject = null;
    $("camPlaceholder").classList.remove("hidden");
    $("pill").className = "pill hidden";
    $("btnStart").textContent = "▶ Mula Kamera";
    $("btnStart").classList.remove("stop");
    $("btnFlip").disabled = true;
  }

  async function loop() {
    if (!running) return;
    const v = $("video");
    if (!busy && v.readyState >= 2) {
      busy = true;
      try {
        // kotak panduan = 80% bahagian tengah
        const res = await predict([drawCrop(v, v.videoWidth, v.videoHeight, 0.8)]);
        if (!ema) ema = res;
        else {
          ema.prob = ema.prob.map((x, i) => x * (1 - SMOOTH) + res.prob[i] * SMOOTH);
          ema.sim = ema.sim.map((x, i) => x * (1 - SMOOTH) + res.sim[i] * SMOOTH);
        }
        if (running) render(ema);
      } finally { busy = false; }
    }
    setTimeout(() => requestAnimationFrame(loop), 120);
  }

  // ---------- Muat naik ----------
  async function handleFile(file) {
    if (!file || !ready) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = async () => {
      $("preview").src = url;
      $("preview").classList.remove("hidden");
      $("dropText").classList.add("hidden");
      setState("idle", "⏳", "Menganalisis…", "");
      const w = img.naturalWidth, h = img.naturalHeight;
      // Test-time augmentation: potongan tengah penuh + zum 85% + zum 70%
      const res = await predict([drawCrop(img, w, h, 1), drawCrop(img, w, h, 0.85), drawCrop(img, w, h, 0.7)]);
      render(res);
      $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
    };
    img.onerror = () => setState("warn", "⚠️", "Gambar tidak dapat dibaca", "Cuba format JPG atau PNG.");
    img.src = url;
  }

  // ---------- UI ----------
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    const cam = b.dataset.tab === "camera";
    $("panel-camera").classList.toggle("hidden", !cam);
    $("panel-upload").classList.toggle("hidden", cam);
    if (!cam) stopCam();
    if (ready) setIdle("Sedia", cam ? "Mulakan kamera untuk mengimbas" : "Pilih gambar untuk dianalisis");
  }));
  $("btnStart").addEventListener("click", () => (running ? stopCam() : startCam()));
  $("btnFlip").addEventListener("click", async () => {
    facing = facing === "environment" ? "user" : "environment";
    stopCam(); await startCam();
  });
  $("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
  // Ambil gambar dengan aplikasi kamera telefon (berfungsi walaupun kamera langsung disekat)
  $("snap").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    stopCam();
    document.querySelector('[data-tab=upload]').click();
    handleFile(f);
    e.target.value = "";
  });
  $("openTab").href = location.href;
  if (inIframe) $("openTab").classList.remove("hidden");
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

  window.__racun = { predictFile: handleFile, decide, get ready() { return ready; }, get nResults() { return nResults; } };
  init();
})();
