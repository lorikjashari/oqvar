// A five-second film of the 3D model on screen: the plan, the structure rising, the finished building.
// Every frame is rendered on its own (none dropped, however slow the GPU) and encoded in the browser
// (WebCodecs H.264, packed as MP4; WebM through MediaRecorder where WebCodecs is missing).
// Walls, openings and rooms are the drawing's own geometry; the heights are the viewer's stated assumptions and
// the finished colours are illustrative for drawings that carry no materials. The film says so in its corner note.
import * as THREE from "three";

const FPS = 30, SECONDS = 5, W = 1920, H = 1080;
const MUXER = "https://cdn.jsdelivr.net/npm/mp4-muxer@5.2.2/build/mp4-muxer.mjs";
const BAR = Math.round((H * (1 - W / H / 2.39)) / 2);  // letterbox bars: a 2.39:1 picture inside 16:9
const ACTS = [["01", "Plan", 0, 0.9], ["02", "Structure", 0.9, 2.7], ["03", "Finish", 2.7, 3.9]];

const ease = (k) => (k <= 0 ? 0 : k >= 1 ? 1 : k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const sine = (k) => (k <= 0 ? 0 : k >= 1 ? 1 : -(Math.cos(Math.PI * k) - 1) / 2);
const span = (t, a, b) => Math.min(Math.max((t - a) / (b - a), 0), 1);
const lerp = (a, b, k) => a + (b - a) * k;
const C = (x) => new THREE.Color(x);

const BLUE = C(0x3f7fe0), CONCRETE = C(0xb4b0a7), LINE = C(0x30363f);
// the light and the world around the model, in three stops: blueprint night, a plain working day, a late-afternoon sun
const LOOK = [
  { at: 0.9, top: C(0x070d18), mid: C(0x0d1830), bottom: C(0x05080e), ground: C(0x0b1424), haze: C(0x0d1830), sky: C(0x2a4a7a), soil: C(0x0a0f18), hemi: 0.55, sunC: C(0x9fb8ff), sun: 0.35 },
  { at: 2.2, top: C(0x50729c), mid: C(0xbcc4cc), bottom: C(0x3a3a3a), ground: C(0x8a8984), haze: C(0xb6bcc2), sky: C(0xd0d9e4), soil: C(0x5c5952), hemi: 1.05, sunC: C(0xf4f2ee), sun: 1.7 },
  { at: 3.7, top: C(0x5d8cc6), mid: C(0xece6dc), bottom: C(0x4a4743), ground: C(0xbdb8ae), haze: C(0xe4ded4), sky: C(0xdfe8f4), soil: C(0x8a8174), hemi: 0.95, sunC: C(0xffe0b8), sun: 2.8 },
];
// finishes for drawings with no materials of their own (CAD plans, scanned plans): plaster, oak, tile, glass
const FLOOR = { bedroom: 0xc49d71, living: 0xc49d71, corridor: 0xcdb08c, kitchen: 0xd8d4cb, bath: 0xd3d9dc, outdoor: 0xa9a59b,
  stair: 0xa3a19b, storage: 0xbdb7ab, other: 0xcbc5b8 };

function finishOf(m, drawn) {
  const u = m.userData, g = u.group, base = u.baseColor ?? m.material.color, op = u.baseOpacity ?? m.material.opacity ?? 1;
  if (!drawn) return { color: base.clone(), opacity: op, rough: m.material.roughness ?? 0.85, metal: 0 };
  if (g === "rooms") return { color: C(FLOOR[u.kind] ?? FLOOR.other), opacity: 1, rough: 0.7, metal: 0 };
  if (g === "windows" && !u.wallLike) return { color: C(0xa9cbe0), opacity: 0.38, rough: 0.05, metal: 0.3 };
  if (g === "doors" && !u.wallLike) return { color: C(0x8f6040), opacity: 1, rough: 0.6, metal: 0 };
  if (g === "fixtures") return { color: C(0xf3f2ee), opacity: 1, rough: 0.4, metal: 0 };
  if (g === "slabs") return { color: C(0xcfcac0), opacity: 1, rough: 0.9, metal: 0 };
  return { color: C(0xefebe4), opacity: 1, rough: 0.85, metal: 0 };   // walls, lintels, parapets
}
// what arrives with the fit-out, after the structure: door leaves, glass, fixtures, furniture
const isLate = (u, drawn) => (drawn
  ? u.group === "fixtures" || (!u.wallLike && (u.group === "doors" || u.group === "windows"))
  : !u.wallLike && ["doors", "windows", "furniture", "sanitary"].includes(u.group));

// ------------------------------------------------------------------ encoders
async function mp4Encoder() {
  if (!("VideoEncoder" in window)) return null;
  const { Muxer, ArrayBufferTarget } = await import(MUXER);
  for (const [codec, kind, label] of [["avc1.640028", "avc", "H.264"], ["vp09.00.40.08", "vp9", "VP9"]]) {
    const cfg = { codec, width: W, height: H, bitrate: 14e6, framerate: FPS };
    let ok = false;
    try { ok = (await VideoEncoder.isConfigSupported(cfg)).supported; } catch { /* try the next codec */ }
    if (!ok) continue;
    const target = new ArrayBufferTarget();
    const muxer = new Muxer({ target, video: { codec: kind, width: W, height: H, frameRate: FPS }, fastStart: "in-memory" });
    let failed = null;
    const enc = new VideoEncoder({ output: (chunk, meta) => muxer.addVideoChunk(chunk, meta), error: (e) => { failed = e; } });
    enc.configure(cfg);
    return {
      ext: "mp4", label: `MP4 · ${label} · ${W}×${H} · ${FPS} fps`,
      async add(canvas, i) {
        if (failed) throw failed;
        const frame = new VideoFrame(canvas, { timestamp: Math.round((i * 1e6) / FPS), duration: Math.round(1e6 / FPS) });
        enc.encode(frame, { keyFrame: i % FPS === 0 }); frame.close();
        while (enc.encodeQueueSize > 4) await new Promise((r) => setTimeout(r, 4));
      },
      async finish() { await enc.flush(); if (failed) throw failed; muxer.finalize(); enc.close(); return new Blob([target.buffer], { type: "video/mp4" }); },
      cancel() { try { enc.close(); } catch { /* already closed */ } },
    };
  }
  return null;
}

function webmEncoder(canvas) {  // real-time pacing: MediaRecorder stamps frames with the wall clock
  const track = canvas.captureStream(0).getVideoTracks()[0];
  const type = ["video/webm;codecs=vp9", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
  const rec = new MediaRecorder(track ? new MediaStream([track]) : canvas.captureStream(0), { mimeType: type, videoBitsPerSecond: 12e6 }), parts = [];
  rec.ondataavailable = (e) => { if (e.data.size) parts.push(e.data); };
  rec.start();
  const t0 = performance.now();
  return {
    ext: "webm", label: `WebM · ${W}×${H} (this browser cannot write MP4)`,
    async add(_, i) { track.requestFrame(); const due = t0 + ((i + 1) * 1000) / FPS; await new Promise((r) => setTimeout(r, Math.max(0, due - performance.now()))); },
    finish: () => new Promise((r) => { rec.onstop = () => r(new Blob(parts, { type: "video/webm" })); rec.stop(); }),
    cancel() { try { rec.stop(); } catch { /* already stopped */ } },
  };
}

// ------------------------------------------------------------------ the set: materials, light, sky, ground, camera
function stage(env) {
  const { renderer, scene, camera, controls, root, box, drawn } = env;
  const keep = {
    tone: renderer.toneMapping, exposure: renderer.toneMappingExposure, shadows: renderer.shadowMap.enabled, clipping: renderer.localClippingEnabled,
    background: scene.background, fog: scene.fog, pos: camera.position.clone(), quat: camera.quaternion.clone(), target: controls.target.clone(),
    fov: camera.fov, near: camera.near, far: camera.far, aspect: camera.aspect, pr: renderer.getPixelRatio(),
    lights: scene.children.filter((o) => o.isLight && o.visible), meshes: [],
  };
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const S = Math.max(size.x, size.z, 4), floorY = box.min.y, topY = box.max.y;
  const clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), topY + 10);  // keeps everything below its height

  const items = [];
  for (const m of root.children) {
    if (!m.isMesh) continue;
    const lines = m.children.filter((o) => o.isLineSegments);
    keep.meshes.push({ m, material: m.material, visible: m.visible, cast: m.castShadow, recv: m.receiveShadow, lines: lines.map((l) => [l, l.material]) });
    if (!m.visible) continue;
    if (!drawn && m.userData.group === "rooms") { m.visible = false; continue; }   // IFC spaces are volumes of air
    const late = isLate(m.userData, drawn), fin = finishOf(m, drawn);
    const mat = new THREE.MeshStandardMaterial({ color: BLUE.clone(), roughness: 0.9, metalness: 0, transparent: true, opacity: 0,
      side: late ? m.material.side : THREE.DoubleSide, flatShading: !!m.material.flatShading, clippingPlanes: [clip], clipShadows: true });
    m.material = mat; m.castShadow = true; m.receiveShadow = true;
    const lm = lines.map((l) => (l.material = new THREE.LineBasicMaterial({ color: BLUE.clone(), transparent: true, opacity: 0, clippingPlanes: [clip] })));
    const g = m.userData.group;
    items.push({ mat, lm, late, fin, plan: g === "rooms" ? 0.16 : g === "slabs" ? 0.1 : 0.95 });
  }

  renderer.localClippingEnabled = true;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  for (const l of keep.lights) l.visible = false;
  const set = new THREE.Group(); scene.add(set);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 1); set.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1);
  sun.position.set(c.x - S * 0.9, floorY + S * 1.1, c.z + S * 0.6); sun.target.position.copy(c);
  sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
  Object.assign(sun.shadow.camera, { left: -S, right: S, top: S, bottom: -S, near: 0.1, far: S * 5 }); sun.shadow.camera.updateProjectionMatrix();
  set.add(sun, sun.target);
  const sky = new THREE.Mesh(new THREE.SphereGeometry(S * 40, 32, 16), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: { top: { value: C(0) }, mid: { value: C(0) }, bottom: { value: C(0) } },
    vertexShader: "varying vec3 vP; void main() { vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: "uniform vec3 top; uniform vec3 mid; uniform vec3 bottom; varying vec3 vP; void main() { float y = vP.y;" +
      " vec3 c = y > 0.0 ? mix(mid, top, smoothstep(0.0, 0.55, y)) : mix(mid, bottom, smoothstep(0.0, 0.2, -y)); gl_FragColor = vec4(c, 1.0); }",
  }));
  sky.position.copy(c); set.add(sky);
  const ground = new THREE.Mesh(new THREE.CircleGeometry(S * 12, 64), new THREE.MeshStandardMaterial({ color: 0x0b1424, roughness: 1 }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(c.x, floorY - 0.14, c.z); ground.receiveShadow = true; set.add(ground);
  const grid = new THREE.GridHelper(S * 3, 48, BLUE, BLUE);
  grid.material.transparent = true; grid.position.set(c.x, floorY - 0.13, c.z); set.add(grid);
  scene.background = null;
  scene.fog = new THREE.Fog(0x0d1830, S * 2.5, S * 9);

  // camera: an orbit that starts high over the plan and lands low on the finished building, ending near the
  // user's own view; the distance is the farthest any pose of the path needs to keep the model inside the 2.39:1 picture
  camera.fov = 35; camera.aspect = W / H; camera.near = S / 400; camera.far = S * 100; camera.updateProjectionMatrix();
  const target = new THREE.Vector3(c.x, lerp(floorY, topY, 0.4), c.z);
  const a1 = Math.atan2(keep.pos.x - c.x, keep.pos.z - c.z), a0 = a1 - 1.25;
  const dirAt = (t) => {
    const az = lerp(a0, a1, sine(t / SECONDS)), el = lerp(1.15, 0.3, ease(span(t, 0, 4.2)));
    return new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az));
  };
  const zoomAt = (t) => lerp(1.12, 0.96, sine(t / SECONDS));   // a slow push in
  const corners = [];
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new THREE.Vector3(x, y, z));
  const need = (dir) => {  // distance at which the box fills 82% of the width and 62% of the height (the bars take the rest)
    let d = size.length();
    for (let i = 0; i < 5; i++) {
      camera.position.copy(target).addScaledVector(dir, d); camera.lookAt(target); camera.updateMatrixWorld();
      let m = 0;
      for (const p of corners) { const v = p.clone().project(camera); if (v.z < 1) m = Math.max(m, Math.abs(v.x) / 0.8, Math.abs(v.y) / 0.6); }
      if (!m) break;
      d *= m;
    }
    return d;
  };
  // the distance each pose needs, sampled every 0.1 s and eased between samples: the model stays large all along
  const STEP = 0.1, dists = [];
  for (let t = 0; t <= SECONDS + 1e-6; t += STEP) dists.push(need(dirAt(t)));
  const distAt = (t) => { const i = Math.min(Math.floor(t / STEP), dists.length - 2), k = t / STEP - i; return lerp(dists[i], dists[i + 1], k); };

  const look = (t, key, out) => {  // the look at time t, eased between its stops
    const i = t <= LOOK[1].at ? 0 : 1, a = LOOK[i], b = LOOK[i + 1], k = ease(span(t, a.at, b.at));
    return out ? out.copy(a[key]).lerp(b[key], k) : lerp(a[key], b[key], k);
  };
  function pose(t) {
    const kPlan = ease(span(t, 0.05, 0.85)), kBlue = ease(span(t, 0.9, 1.6)), kBuild = ease(span(t, 0.9, 2.7)), kFin = ease(span(t, 2.7, 3.9));
    clip.constant = kBuild >= 1 ? topY + 10 : t < 0.9 ? floorY + 0.15 : lerp(floorY + 0.15, topY + 0.05, kBuild);
    for (const it of items) {
      const m = it.mat;
      if (it.late) {
        m.color.copy(it.fin.color); m.opacity = it.fin.opacity * kFin; m.roughness = it.fin.rough; m.metalness = it.fin.metal;
      } else {
        m.color.copy(BLUE).lerp(CONCRETE, kBlue).lerp(it.fin.color, kFin);
        m.emissive.copy(BLUE).multiplyScalar(0.55 * (1 - kBlue));
        m.opacity = t < 0.9 ? it.plan * kPlan : lerp(lerp(it.plan, 1, kBlue), it.fin.opacity, kFin);
        m.roughness = lerp(0.9, it.fin.rough, kFin); m.metalness = lerp(0, it.fin.metal, kFin);
      }
      m.depthWrite = m.opacity > 0.6; m.visible = m.opacity > 0.003;
      for (const l of it.lm) { l.color.copy(BLUE).lerp(LINE, kBlue); l.opacity = it.late ? 0.16 * kFin : t < 0.9 ? 0.95 * kPlan : lerp(0.55, 0.16, kFin); }
    }
    const u = sky.material.uniforms;
    look(t, "top", u.top.value); look(t, "mid", u.mid.value); look(t, "bottom", u.bottom.value);
    look(t, "haze", scene.fog.color); look(t, "ground", ground.material.color);
    grid.material.opacity = 0.5 * kPlan * (1 - ease(span(t, 1.4, 3.2)));
    look(t, "sky", hemi.color); look(t, "soil", hemi.groundColor); hemi.intensity = look(t, "hemi");
    look(t, "sunC", sun.color); sun.intensity = look(t, "sun");
    camera.position.copy(target).addScaledVector(dirAt(t), distAt(t) * zoomAt(t)); camera.lookAt(target);
  }

  function restore() {
    scene.remove(set); set.traverse((o) => { o.geometry?.dispose(); o.material?.dispose?.(); });
    for (const k of keep.meshes) {
      if (k.m.material !== k.material) k.m.material.dispose();
      Object.assign(k.m, { material: k.material, visible: k.visible, castShadow: k.cast, receiveShadow: k.recv });
      for (const [l, mat] of k.lines) { if (l.material !== mat) l.material.dispose(); l.material = mat; }
    }
    for (const l of keep.lights) l.visible = true;
    Object.assign(renderer, { toneMapping: keep.tone, toneMappingExposure: keep.exposure, localClippingEnabled: keep.clipping });
    renderer.shadowMap.enabled = keep.shadows;
    scene.background = keep.background; scene.fog = keep.fog;
    camera.position.copy(keep.pos); camera.quaternion.copy(keep.quat); controls.target.copy(keep.target);
    Object.assign(camera, { fov: keep.fov, near: keep.near, far: keep.far, aspect: keep.aspect }); camera.updateProjectionMatrix();
    renderer.setPixelRatio(keep.pr);
  }
  return { pose, restore };
}

// ------------------------------------------------------------------ titles over the picture
async function loadFonts() {
  if (!document.querySelector("link[data-film-fonts]")) {
    document.head.append(Object.assign(document.createElement("link"), { rel: "stylesheet", href: "/web/fonts/fonts.css" }));
    document.head.lastChild.dataset.filmFonts = "";
  }
  try { await Promise.all(["700 30px Unbounded", "600 54px Sora", "500 24px 'IBM Plex Sans'", "500 17px 'IBM Plex Mono'"].map((f) => document.fonts.load(f))); }
  catch { /* system fonts stand in */ }
}

function titles(g, t, env) {
  const bone = (a) => `rgba(236, 230, 214, ${a})`, pad = 72;
  const vig = g.createRadialGradient(W / 2, H / 2, H * 0.45, W / 2, H / 2, W * 0.62);   // a soft vignette
  vig.addColorStop(0, "rgba(0, 0, 0, 0)"); vig.addColorStop(1, "rgba(0, 0, 0, 0.32)");
  g.fillStyle = vig; g.fillRect(0, 0, W, H);
  // the end title sits on a soft shade at the foot of the picture
  const kT = ease(span(t, 3.85, 4.55));
  if (kT > 0) {
    const sh = g.createLinearGradient(0, H - BAR - 300, 0, H - BAR);
    sh.addColorStop(0, "rgba(5, 6, 8, 0)"); sh.addColorStop(1, `rgba(5, 6, 8, ${0.62 * kT})`);
    g.fillStyle = sh; g.fillRect(0, H - BAR - 300, W, 300);
    g.globalAlpha = kT; g.textAlign = "left"; g.textBaseline = "alphabetic"; g.letterSpacing = "0px";
    const y = H - BAR - 64 + (1 - kT) * 16;
    g.font = "600 54px Sora, 'IBM Plex Sans', sans-serif"; g.fillStyle = bone(1); g.fillText(env.title, pad, y, W - 2 * pad);
    g.font = "500 20px 'IBM Plex Mono', monospace"; g.letterSpacing = "1px"; g.fillStyle = bone(0.62); g.fillText(env.source, pad, y + 40, W - 2 * pad);
    g.globalAlpha = 1;
  }
  g.fillStyle = "#050608"; g.fillRect(0, 0, W, BAR); g.fillRect(0, H - BAR, W, BAR);
  g.textBaseline = "middle";
  g.font = "700 30px Unbounded, sans-serif"; g.letterSpacing = "3px"; g.textAlign = "left"; g.fillStyle = bone(1); g.fillText("OQVAR", pad, BAR / 2);
  g.font = "500 17px 'IBM Plex Mono', monospace"; g.letterSpacing = "4px"; g.textAlign = "right"; g.fillStyle = bone(0.6); g.fillText("FROM PLAN TO BUILDING", W - pad, BAR / 2);
  // the three acts, each with a rule that fills while it plays
  let x = pad; const y = H - BAR / 2 - 6;
  g.textAlign = "left";
  for (const [n, name, a, b] of ACTS) {
    const on = t >= a && (t < b || name === "Finish"), done = t >= b && name !== "Finish";
    g.font = "500 17px 'IBM Plex Mono', monospace"; g.letterSpacing = "2px"; g.fillStyle = on ? "#ff5a4a" : bone(done ? 0.55 : 0.3); g.fillText(n, x, y);
    const w1 = g.measureText(n).width + 14;
    g.font = "500 24px 'IBM Plex Sans', sans-serif"; g.letterSpacing = "0px"; g.fillStyle = bone(on ? 1 : done ? 0.55 : 0.3); g.fillText(name, x + w1, y);
    const w = w1 + g.measureText(name).width;
    g.fillStyle = bone(0.14); g.fillRect(x, y + 26, w, 2);
    g.fillStyle = on ? "#ff5a4a" : bone(0.55); g.fillRect(x, y + 26, w * span(t, a, b), 2);
    x += w + 56;
  }
  g.textAlign = "right"; g.font = "400 16px 'IBM Plex Sans', sans-serif"; g.letterSpacing = "0px"; g.fillStyle = bone(0.5);
  g.fillText(env.note, W - pad, y - 10, W / 2 - pad);
  if (env.credit) { g.fillStyle = bone(0.34); g.font = "400 14px 'IBM Plex Sans', sans-serif"; g.fillText(env.credit, W - pad, y + 16, W / 2 - pad); }
  const kIn = 1 - ease(span(t, 0, 0.5));   // fade up from black
  if (kIn > 0) { g.fillStyle = `rgba(0, 0, 0, ${kIn})`; g.fillRect(0, 0, W, H); }
}

// ------------------------------------------------------------------ the dialog: live preview while filming, then play / download
const CSS = `
.film { position: absolute; inset: 0; z-index: 20; display: grid; place-items: center; padding: 20px; background: rgba(6, 7, 9, .72); backdrop-filter: blur(6px); }
.film-card { width: min(560px, 100%); padding: 14px; border: 1px solid rgba(236, 230, 214, .18); border-radius: 16px; background: rgba(20, 22, 27, .96);
  box-shadow: 0 30px 80px -20px rgba(0, 0, 0, .9); color: #ece6d6; font: 13px/1.45 "IBM Plex Sans", system-ui, sans-serif; }
.film-card canvas, .film-card video { display: block; width: 100%; aspect-ratio: 16 / 9; border-radius: 10px; background: #000; }
.film-bar { height: 4px; margin: 12px 0 8px; border-radius: 2px; background: rgba(236, 230, 214, .12); overflow: hidden; }
.film-bar i { display: block; height: 100%; width: 0; background: #ff5a4a; transition: width .15s; }
.film-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-top: 12px; }
.film-row p { margin: 0; color: rgba(236, 230, 214, .62); font-size: 12.5px; }
.film-row b { color: #ece6d6; font-weight: 600; }
.film-acts { display: flex; gap: 8px; flex: none; }
.film-btn { display: inline-flex; align-items: center; height: 32px; padding: 0 14px; border: 0; border-radius: 16px; background: rgba(236, 230, 214, .1);
  color: #ece6d6; font: 600 12.5px "IBM Plex Sans", system-ui, sans-serif; text-decoration: none; cursor: pointer; }
.film-btn:hover { background: rgba(236, 230, 214, .18); }
.film-btn.go { background: #ece6d6; color: #08090b; }
.film-btn.go:hover { opacity: .9; }`;

function dialog(pane) {
  if (!document.getElementById("film-css")) document.head.append(Object.assign(document.createElement("style"), { id: "film-css", textContent: CSS }));
  const root = document.createElement("div");
  root.className = "film"; root.setAttribute("role", "dialog"); root.setAttribute("aria-label", "Film");
  root.innerHTML = `<div class="film-card"><canvas width="640" height="360"></canvas><div class="film-bar"><i></i></div>
    <div class="film-row"><p aria-live="polite">Setting the scene…</p><div class="film-acts"><button class="film-btn" data-act="cancel">Cancel</button></div></div></div>`;
  pane.append(root);
  return root;
}

// env: { renderer, scene, camera, controls, root, box, drawn, pane, title, source, note, credit, name }
export async function makeFilm(env) {
  const ui = dialog(env.pane), prev = ui.querySelector("canvas").getContext("2d"), msg = ui.querySelector("p"), bar = ui.querySelector(".film-bar i");
  let cancelled = false;
  ui.querySelector("[data-act=cancel]").onclick = () => { cancelled = true; };
  const out = Object.assign(document.createElement("canvas"), { width: W, height: H }), g = out.getContext("2d");
  const set = stage(env);
  let enc = null, blob = null, error = null;
  try {
    await loadFonts();
    enc = (await mp4Encoder().catch(() => null)) ?? webmEncoder(out);
    env.renderer.setPixelRatio(1); env.renderer.setSize(W, H, false);
    const n = FPS * SECONDS;
    for (let i = 0; i < n && !cancelled; i++) {
      const t = i / FPS;
      set.pose(t); env.renderer.render(env.scene, env.camera);
      g.drawImage(env.renderer.domElement, 0, 0, W, H); titles(g, t, env);
      await enc.add(out, i);
      if (i % 2 === 0 || i === n - 1) {
        prev.drawImage(out, 0, 0, 640, 360); bar.style.width = `${((i + 1) / n) * 100}%`;
        msg.innerHTML = `<b>Filming</b> · ${ACTS.find(([, , a, b]) => t >= a && t < b)?.[1] ?? "Finish"} · frame ${i + 1} of ${n}`;
        await new Promise((r) => setTimeout(r));
      }
    }
    if (!cancelled) { msg.innerHTML = "<b>Packing the video</b>…"; blob = await enc.finish(); }
    else enc.cancel();
  } catch (e) { error = e; enc?.cancel(); }
  finally { set.restore(); env.after?.(); }
  if (cancelled) { ui.remove(); return null; }
  if (error || !blob) {
    msg.innerHTML = `<b>The film could not be made</b> · ${String(error?.message || error || "no video was written")}`;
    ui.querySelector(".film-acts").innerHTML = `<button class="film-btn" data-act="close">Close</button>`;
    ui.querySelector("[data-act=close]").onclick = () => ui.remove();
    return null;
  }
  const url = URL.createObjectURL(blob), poster = await new Promise((r) => out.toBlob(r, "image/png")), purl = URL.createObjectURL(poster);
  const file = `${(env.name || "drawing").replace(/[^\w.-]+/g, "_")}-plan-to-building`;
  const card = ui.querySelector(".film-card");
  card.querySelector("canvas").replaceWith(Object.assign(document.createElement("video"), { src: url, autoplay: true, loop: true, muted: true, playsInline: true, controls: true }));
  bar.style.width = "100%";
  msg.innerHTML = `<b>5-second film</b> · ${enc.label} · ${(blob.size / 1e6).toFixed(1)} MB`;
  ui.querySelector(".film-acts").innerHTML = `<button class="film-btn" data-act="close">Close</button>
    <a class="film-btn" href="${purl}" download="${file}.png" title="The last frame as a picture">Poster</a>
    <a class="film-btn go" href="${url}" download="${file}.${enc.ext}">Download ${enc.ext.toUpperCase()}</a>`;
  ui.querySelector("[data-act=close]").onclick = () => { ui.remove(); URL.revokeObjectURL(url); URL.revokeObjectURL(purl); };
  return blob;
}
