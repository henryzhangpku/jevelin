// Runs the jevelin Python engine under Pyodide, off the main thread.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const MODULES = ["__init__", "backends", "calls", "executor", "pack",
                 "pipelines", "policy", "profile", "report", "web"];
let ready = null;

async function boot(base) {
  const py = await loadPyodide();
  py.FS.mkdirTree("/home/pyodide/jevelin");
  const get = async (path) => {
    const res = await fetch(base + path);
    if (!res.ok) throw new Error("Could not load " + path + " (" + res.status + ")");
    return res.text();
  };
  for (const m of MODULES) {
    py.FS.writeFile("/home/pyodide/jevelin/" + m + ".py", await get("jevelin/" + m + ".py"));
  }
  py.globals.set("PACK_RAW", await get("packs/support.json"));
  py.globals.set("PROFILE_RAW", await get("profiles/default.json"));
  await py.runPythonAsync(`
import json, sys
sys.path.insert(0, "/home/pyodide")
from jevelin.web import simulate
PACK = json.loads(PACK_RAW)
PROFILE = json.loads(PROFILE_RAW)
`);
  return py;
}

self.onmessage = async (e) => {
  const { type, base, settings, id } = e.data;
  try {
    if (type === "boot") {
      ready = boot(base);
      await ready;
      self.postMessage({ type: "ready" });
      return;
    }
    const py = await ready;
    py.globals.set("SETTINGS", JSON.stringify(settings));
    const t0 = performance.now();
    const out = await py.runPythonAsync(
      "json.dumps(await simulate(PACK, PROFILE, json.loads(SETTINGS)))");
    self.postMessage({ type: "result", id, data: JSON.parse(out), ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ type: "error", id, message: String(err && err.message || err) });
  }
};
