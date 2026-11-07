const $ = id => document.getElementById(id);
const examples = {
  numbered: "0. The studio door is blue.\n1. The meeting starts at nine.\n2. The studio door is green.\n3. The report is in the shared folder.\n4. The meeting starts at ten.\n5. The studio door is white.",
  serial: "Step 0: The studio door is blue.\nStep 1: The meeting starts at nine.\nStep 2: The studio door is green.\nStep 3: The report is in the shared folder.\nStep 4: The meeting starts at ten.\nStep 5: The studio door is white.",
  records: "Session A\nThe team tested a small robot in the lab. It recorded wheel speed and camera timestamps.\n\nSession B\nThe team added a new sensor and kept the original measurements for comparison.\n\nSession C\nThe report separates decisions, measurements and open questions.",
  prose: "The river curved through the valley as the survey team followed its banks. They described the landscape in continuous prose, connecting each observation to the next rather than separating it into individual records. ".repeat(160)
};
$("store").value = examples.numbered;
let requestId = 0;
let worker = null;
let busy = false;
let slowTimer;
function setBusy(value) {
  busy = value;
  $("inspect").disabled = value;
  $("cancel").hidden = !value;
  $("store").setAttribute("aria-busy", String(value));
  if (!value) clearTimeout(slowTimer);
}
function stopInspection(message) {
  ++requestId;
  if (worker) worker.terminate();
  worker = null;
  setBusy(false);
  window.lastInspectedPlan = null;
  $("result-panel").hidden = true;
  $("status").textContent = message;
}
function createWorker() {
  const next = new Worker(new URL("inspect.bundle.js", import.meta.url), {type: "module"});
  next.onmessage = event => { if (next === worker) handleMessage(event); };
  next.onerror = () => {
    if (next !== worker) return;
    stopInspection("The local tokenizer stopped. Try Inspect structure again, or use the Python SDK.");
  };
  worker = next;
}
function run() {
  const text = $("store").value;
  if (text.length > 200000) { $("status").textContent = "Use up to 200,000 characters for this preview, or the Python SDK for a larger store. No text was truncated."; return; }
  if (busy) stopInspection("");
  if (!worker) createWorker();
  window.lastInspectedPlan = null;
  $("result-panel").hidden = true;
  setBusy(true);
  $("status").textContent = "Reading structure locally…";
  slowTimer = setTimeout(() => {
    if (busy) $("status").textContent = "Still counting tokens locally. Large inputs can take a while; Cancel keeps your text.";
  }, 1500);
  worker.postMessage({id: ++requestId, text});
}
function handleMessage(event) {
  if (event.data.id !== requestId) return;
  setBusy(false);
  if (event.data.error) { $("status").textContent = event.data.error; return; }
  const plan = event.data.result;
  window.lastInspectedPlan = plan;
  $("result-panel").hidden = false;
  $("branch").textContent = {numbered:"Numbered records",record:"Record store",prose:"Continuous prose",empty:"Empty store"}[plan.branch];
  $("serial").textContent = (plan.stats.serial * 100).toFixed(1) + "%";
  $("record").textContent = (plan.stats.record_share * 100).toFixed(1) + "%";
  $("unit").textContent = plan.branch === "numbered" ? "One record" : plan.chunk ? plan.chunk.toLocaleString() + " tokens" : "—";
  $("read").textContent = String(plan.U);
  $("parent").textContent = plan.branch === "numbered" ? "By derived slot" : plan.mscale ? plan.mscale.toLocaleString() + " tokens" : "No parent layer";
  $("policy").textContent = plan.branch === "empty" ? "—" : (plan.branch === "numbered" ? "2 : 0 : 8" : "4 : 3 : 3") + " · dense / lexical / parent";
  $("explanation").textContent = plan.branch === "numbered" ? (plan.serial === "lead" ? "The short lines count up one at a time in the log's own style: at least 95% of adjacent lines advance by one and the median line uses at most 20 tokens. Construction uses record units and derived slots; order-based replacement applies where a slot is found." : "At least 80% of nonempty lines use the integer-dot format. Construction uses record units and derived slots; order-based replacement applies where a slot is found.") : plan.branch === "record" ? "At least 90% of the record tokens fit within the reference chunk. Keep record boundaries when packing units and use balanced retrieval floors." : plan.branch === "prose" ? "Short records cover less than 90% of the text. Search with finer prose units and more units per read; preserve the source text." : "Add text to inspect its structure.";
  $("json").textContent = JSON.stringify(plan, null, 2);
  $("status").textContent = plan.stats.tokens.toLocaleString() + " tokens · " + plan.stats.lines_used.toLocaleString() + " nonempty lines inspected · nothing uploaded";
}
$("inspect").addEventListener("click", run);
$("cancel").addEventListener("click", () => {
  stopInspection("Inspection cancelled. Your text is unchanged.");
  $("inspect").focus();
});
$("store").addEventListener("input", () => {
  if (busy) stopInspection("Text changed. Inspect it again to update the configuration.");
  else {
    window.lastInspectedPlan = null;
    $("result-panel").hidden = true;
    $("status").textContent = "Text changed. Inspect it again to update the configuration.";
  }
});
document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => { $("store").value = examples[button.dataset.example]; run(); }));
// Worker initialization and tokenization are queued off the UI thread.
run();
