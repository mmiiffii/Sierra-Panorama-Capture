
// docs/app.js
(function () {
  const cfg = window.SN_CONFIG;
  const RAW = (path) =>
    `https://raw.githubusercontent.com/${cfg.owner}/${cfg.repo}/${cfg.branch}/${path}`;
  const API_CONTENTS = (path) =>
    `https://api.github.com/repos/${cfg.owner}/${cfg.repo}/contents/${path}?ref=${cfg.branch}`;

  const qs = (s, r = document) => r.querySelector(s);

  function byId(id) { return cfg.cameras.find(c => c.id === id); }

  async function listImages(camId) {
    const cam = byId(camId);
    if (!cam) throw new Error("Unknown camera: " + camId);
    const res = await fetch(API_CONTENTS(cam.dir), { headers: { 'Accept': 'application/vnd.github+json' } });
    if (!res.ok) {
      console.warn("GitHub API error:", res.status, await res.text());
      return [];
    }
    const data = await res.json();
    if (!Array.isArray(data)) return [];
    const files = data
      .filter(x => x.type === "file" && /\.jpe?g$/i.test(x.name))
      .map(x => ({ name: x.name, path: `${cam.dir}/${x.name}` }));
    files.sort((a, b) => a.name.localeCompare(b.name));
    return files;
  }

  function parseTsFromName(name, camId) {
    const re = new RegExp(`^${camId}_(\\d{6})_(\\d{6})`);
    const m = name.match(re);
    if (!m) return null;
    const d = m[1], t = m[2];
    const y = 2000 + parseInt(d.slice(0,2), 10);
    const mo = parseInt(d.slice(2,4), 10) - 1;
    const da = parseInt(d.slice(4,6), 10);
    const hh = parseInt(t.slice(0,2), 10);
    const mm = parseInt(t.slice(2,4), 10);
    const ss = parseInt(t.slice(4,6), 10);
    return new Date(Date.UTC(y, mo, da, hh, mm, ss));
  }

  function fmt(dt) {
    if (!dt) return "";
    const pad = (n) => String(n).padStart(2, "0");
    return `${dt.getUTCFullYear()}-${pad(dt.getUTCMonth()+1)}-${pad(dt.getUTCDate())} ${pad(dt.getUTCHours())}:${pad(dt.getUTCMinutes())}:${pad(dt.getUTCSeconds())} UTC`;
  }

  function cameraListHTML(activeId) {
    return cfg.cameras.map(c => {
      const href = `./camera.html?cam=${encodeURIComponent(c.id)}`;
      const cls = c.id === activeId ? "cam-link active" : "cam-link";
      return `<a class="${cls}" href="${href}">${c.name}</a>`;
    }).join("");
  }

  function renderMapPins(activeId) {
    const container = document.getElementById("map-markers");
    if (!container) return;
    container.innerHTML = "";
    cfg.cameras.forEach(c => {
      const el = document.createElement("div");
      el.className = "map-pin" + (c.id === activeId ? " active" : "");
      const [x, y] = c.mapPct || [0.5, 0.5];
      el.style.left = (x * 100) + "%";
      el.style.top = (y * 100) + "%";
      el.title = c.name;
      el.addEventListener("click", () => {
        window.location.href = `./camera.html?cam=${encodeURIComponent(c.id)}`;
      });
      container.appendChild(el);
    });
  }

  async function buildIndex() {
    document.getElementById("cam-list").innerHTML = cameraListHTML(null);
    renderMapPins(null);

    const grid = document.getElementById("card-grid");
    grid.innerHTML = "";

    for (const cam of cfg.cameras) {
      const files = await listImages(cam.id);
      const last = files[files.length - 1];
      if (!last) continue;
      const url = RAW(last.path);
      const ts = fmt(parseTsFromName(last.name, cam.id));
      const card = document.createElement("a");
      card.href = `./camera.html?cam=${encodeURIComponent(cam.id)}`;
      card.className = "card";
      card.innerHTML = `<img loading="lazy" src="${url}" alt="${cam.name} latest" />
                        <div class="meta">${cam.name} • ${ts}</div>`;
      grid.appendChild(card);
    }
  }

  async function buildCamera() {
    const params = new URLSearchParams(location.search);
    const camId = params.get("cam") || (cfg.cameras[0]?.id);
    const cam = byId(camId);
    if (!cam) return;

    document.getElementById("cam-title").textContent = cam.name;
    document.getElementById("cam-list").innerHTML = cameraListHTML(camId);
    renderMapPins(camId);

    // selector
    const sel = document.getElementById("cam-select");
    sel.innerHTML = cfg.cameras.map(c => `<option value="${c.id}" ${c.id===camId?"selected":""}>${c.name}</option>`).join("");
    sel.addEventListener("change", e => {
      location.href = `./camera.html?cam=${encodeURIComponent(e.target.value)}`;
    });

    const files = await listImages(camId);
    const thumbs = document.getElementById("thumbs");
    let idxStart = Math.max(0, files.length - cfg.pageSize);
    let currentIdx = files.length - 1;

    function setHero(i) {
      const f = files[i];
      if (!f) return;
      currentIdx = i;
      document.getElementById("hero-img").src = RAW(f.path);
      const dt = parseTsFromName(f.name, camId);
      document.getElementById("hero-meta").textContent = `${cam.name} • ${fmt(dt)}`;
    }

    function renderThumbs(start, endExclusive) {
      thumbs.innerHTML = "";
      for (let i = endExclusive - 1; i >= start; i--) {
        const f = files[i];
        const a = document.createElement("div");
        a.className = "thumb";
        a.innerHTML = `<img loading="lazy" src="${RAW(f.path)}" alt="${f.name}" />`;
        a.addEventListener("click", () => setHero(i));
        thumbs.appendChild(a);
      }
    }

    // initial chunk
    setHero(currentIdx);
    renderThumbs(idxStart, files.length);

    // load more button
    document.getElementById("btn-load-more").addEventListener("click", () => {
      idxStart = Math.max(0, idxStart - cfg.pageSize);
      renderThumbs(idxStart, files.length);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    const isIndex = !!document.getElementById("card-grid");
    if (isIndex) buildIndex().catch(console.error);
    else buildCamera().catch(console.error);
  });
})();
