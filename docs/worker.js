// Runs the jevelin Python engine under Pyodide, off the main thread.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const MODULES = ["__init__", "backends", "calls", "executor", "pack",
                 "pipelines", "policy", "profile", "report", "web"];
let ready = null;
let BASE = "./";

async function boot(base) {
  BASE = base;
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
from jevelin.pack import Pack
PACK = json.loads(PACK_RAW)
PROFILE = json.loads(PROFILE_RAW)
`);
  return py;
}

// Swap the active pack. Either a shipped one by name, or raw JSON from a file
// the viewer chose, which never leaves their machine.
async function setPack(py, { name, json }) {
  let raw = json;
  if (raw === undefined) {
    const res = await fetch(BASE + "packs/" + name + ".json");
    if (!res.ok) throw new Error("No pack called " + name + " (" + res.status + ")");
    raw = await res.text();
  }
  py.globals.set("NEW_PACK_RAW", raw);
  // Pack() validates: every protective flag needs a question, every action and
  // rule target needs a script, hold and safe_fallback must exist. A bad file
  // raises here and the previous pack is left untouched.
  return await py.runPythonAsync(`
_cand = json.loads(NEW_PACK_RAW)
Pack(_cand)
PACK = _cand
PACK.get("name", "pack")
`);
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

    if (type === "setpack") {
      const name = await setPack(py, e.data);
      self.postMessage({ type: "packed", id, name });
      return;
    }

    py.globals.set("SETTINGS", JSON.stringify(settings));
    const t0 = performance.now();
    const out = await py.runPythonAsync(
      "json.dumps(await simulate(PACK, PROFILE, json.loads(SETTINGS)))");
    self.postMessage({ type: "result", id, data: JSON.parse(out), ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({
      type: "error", id, kind: type,
      message: String(err && err.message || err),
    });
  }
};
