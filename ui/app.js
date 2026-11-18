// PathSnap evaluator UI: lists trips from the API, sends the chosen trip ID to
// POST /predict and draws the returned route. It never sees reference routes.
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let API = params.get("api") || localStorage.getItem("pathsnap_api") || "http://localhost:8000";
$("api").value = API;

const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", {
  maxZoom: 19, attribution: "Tiles &copy; Esri &mdash; OpenStreetMap contributors",
}).addTo(map);

const layers = {
  roads: L.layerGroup(), gps: L.layerGroup(), clf: L.layerGroup(), route: L.layerGroup(),
};
layers.roads.addTo(map); layers.gps.addTo(map); layers.route.addTo(map);
L.control.layers(null, {
  "Nearby road network": layers.roads,
  "Raw GPS observations": layers.gps,
  "Classifier-only route": layers.clf,
  "Predicted route (HMM)": layers.route,
}, { collapsed: false }).addTo(map);

const legend = L.control({ position: "bottomright" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = '<i class="dot" style="background:#2563eb"></i>GPS observation<br>' +
    '<i style="background:#16a34a"></i>Predicted route (HMM)<br>' +
    '<i style="background:#f97316"></i>Classifier-only route<br>' +
    '<i style="background:#9ca3af"></i>Nearby roads';
  return d;
};
legend.addTo(map);

function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = "status " + cls;
}

async function connect() {
  API = $("api").value.trim().replace(/\/$/, "");
  try { localStorage.setItem("pathsnap_api", API); } catch (e) { /* storage unavailable */ }
  setStatus("Connecting...");
  try {
    const h = await (await fetch(`${API}/health`)).json();
    const t = await (await fetch(`${API}/trips`)).json();
    setStatus(`Connected: model ${h.model} (${h.params_source}), ${t.trips.length} trips`, "ok");
    const box = $("trips");
    box.innerHTML = "";
    t.trips.forEach((id) => {
      const b = document.createElement("button");
      b.textContent = id;
      b.onclick = () => runTrip(id, b);
      box.appendChild(b);
    });
  } catch (e) {
    setStatus(`Cannot reach ${API}: ${e.message}`, "err");
  }
}

const pct = (v) => (v === null || v === undefined ? "n/a" : (100 * v).toFixed(2) + " %");

async function runTrip(id, btn) {
  document.querySelectorAll(".trips button").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  $("results").innerHTML = `<p class="muted">Running pipeline for trip ${id}...</p>`;
  const t0 = performance.now();
  let r;
  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: id }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    r = await res.json();
  } catch (e) {
    $("results").innerHTML = `<p class="status err">Request failed: ${e.message}</p>`;
    return;
  }
  const rtt = (performance.now() - t0) / 1000;
  draw(r);
  const m = r.metrics;
  $("results").innerHTML = `
    <div class="muted">Point accuracy (HMM)</div>
    <div class="big">${pct(m.hmm_point_accuracy)}</div>
    <table>
      <tr><td>Trip</td><td>${r.trip_id}</td></tr>
      <tr><td>GPS observations</td><td>${r.n_points.toLocaleString()}</td></tr>
      <tr><td>Points scored (reliable reference)</td><td>${m.n_scored_points.toLocaleString()}</td></tr>
      <tr><td>Classifier-only point accuracy</td><td>${pct(m.classifier_point_accuracy)}</td></tr>
      <tr><td>Route precision (length)</td><td>${pct(m.route_precision)}</td></tr>
      <tr><td>Route recall (length)</td><td>${pct(m.route_recall)}</td></tr>
      <tr><td>Route F1</td><td>${pct(m.route_f1)}</td></tr>
      <tr><td>Route mismatch fraction</td><td>${m.route_mismatch_fraction.toFixed(4)}</td></tr>
      <tr><td>Continuity (HMM / classifier)</td><td>${pct(m.hmm_continuity)} / ${pct(m.classifier_continuity)}</td></tr>
      <tr><td>Predicted route length</td><td>${m.pred_route_km.toFixed(2)} km</td></tr>
      <tr><td>Server time</td><td>${r.timings.predict_total_s.toFixed(2)} s</td></tr>
      <tr><td>Round trip</td><td>${rtt.toFixed(2)} s</td></tr>
      <tr><td>Model</td><td>${r.model}</td></tr>
    </table>`;
}

function draw(r) {
  Object.values(layers).forEach((g) => g.clearLayers());
  r.nearby_roads.forEach((seg) => L.polyline(seg, { color: "#9ca3af", weight: 2, opacity: 0.6 }).addTo(layers.roads));
  r.gps.forEach((p, i) => L.circleMarker(p, { radius: 2, color: "#2563eb", weight: 0, fillOpacity: 0.6 })
    .bindTooltip(`GPS #${i}`).addTo(layers.gps));
  r.classifier_route.forEach((line) => L.polyline(line, { color: "#f97316", weight: 4, opacity: 0.8, dashArray: "6 6" }).addTo(layers.clf));
  r.predicted_route.forEach((line) => L.polyline(line, { color: "#16a34a", weight: 4, opacity: 0.85 }).addTo(layers.route));
  const pts = r.gps.length ? L.latLngBounds(r.gps) : null;
  if (pts) map.fitBounds(pts.pad(0.05));
}

$("connect").onclick = connect;
$("api").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
connect();
