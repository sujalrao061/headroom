"use client";

import { useEffect, useState } from "react";
import { ArrowDownRight, ArrowRight, Check, ChevronDown, CircleHelp, Download, Play, ShieldCheck, X } from "lucide-react";

type CheckResult = { shape: string; passed: boolean; expected_shape: number[]; actual_shape: number[] };
type Metrics = { latency_ms: number; peak_allocated_mb: number | null; peak_total_allocated_mb: number | null; base_allocated_mb: number | null; peak_reserved_mb: number | null };
type Result = {
  case_id: string; name: string; contract: string; source: string; hot_line: number; size: number;
  memory_limit_mb: number; latency_limit_ms: number; device: string; torch_version: string;
  cuda_version: string | null; python_version: string; baseline: Metrics; candidate: Metrics; library: Metrics & { contract_passed: boolean; within_limits: boolean };
  checks: CheckResult[]; accepted: boolean; reasons: string[]; repetitions: number; warmups: number;
  failure_kind: string; patch: string; patch_diff: string; regression_test: string; regression_test_source: string; method: string; candidate_origin: string;
  frontier: { tested: { size: number; latency_ms: number; peak_allocated_mb: number | null; within_limits: boolean }[]; largest_tested_within_limits: number | null; untested_sizes: string } | null;
  nemotron: { status: string; model: string | null; strategy: string; explanation: string; usage?: unknown };
  recorded_at?: string;
};
type AttentionResult = {
  shape: { sequence_length: number; head_dimension: number }; device: string; memory_budget_mib: number;
  baseline: Metrics; candidate: Metrics; library: Metrics;
  checks: { shape: number[]; passed: boolean; library_passed: boolean }[];
  baseline_within_budget: boolean; candidate_within_budget: boolean; library_within_budget: boolean;
  recommended: string; model_agrees: boolean | null;
  nemotron: { status: string; strategy: string | null; explanation: string; model: string | null };
  limitations: string;
  recorded_at?: string; evidence_verified?: boolean;
};
type RecordedDecision = { status: "live"; model: string; strategy: string; agrees_with_validator: boolean; evidence_captured_at: string };
type GauntletEvidence = {
  captured_at_utc: string; scope: string; device: string; torch_version: string; python_version: string;
  passed: boolean; checks_passed: number; checks_total: number;
  parameters: { random_shapes: number; seeds_per_shape: number; tile_sizes: number[]; cases: string[]; stress_input_families?: string[] };
  regression_sensitivity: { nearest_that_ignores_later_reference_tiles_detected: boolean; radius_that_excludes_exact_boundary_detected: boolean };
};

const workloads = [
  { id: "nearest", tag: "REPAIRABLE", title: "Nearest neighbor", summary: "Find one closest reference point per query.", contract: "One index per query", formula: "[N] indices" },
  { id: "matrix", tag: "CONTRACT TRAP", title: "Full distance matrix", summary: "Return every pairwise distance, not just the closest.", contract: "Every pairwise distance", formula: "[N × M] values" },
  { id: "radius", tag: "SECOND REPAIR", title: "Radius count", summary: "Count references within radius 1.0 for each query.", contract: "One count per query", formula: "[N] counts" },
];

function saveReport(result: Result) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `headroom-${result.case_id}-${result.size}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function saveRegression(result: Result) {
  const blob = new Blob([result.regression_test_source], { type: "text/x-python" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `test_headroom_${result.case_id}.py`;
  link.click();
  URL.revokeObjectURL(url);
}

export default function Home() {
  const [caseId, setCaseId] = useState("nearest");
  const [size, setSize] = useState(1536);
  const [memory, setMemory] = useState(8);
  const [latency, setLatency] = useState(1000);
  const [result, setResult] = useState<Result | null>(null);
  const [resultSource, setResultSource] = useState<"recorded" | "live">("live");
  const [recordedDecision, setRecordedDecision] = useState<RecordedDecision | null>(null);
  const [decisionBusy, setDecisionBusy] = useState(false);
  const [decisionError, setDecisionError] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [supportedCases, setSupportedCases] = useState<string[]>(["nearest", "matrix", "radius"]);
  const [runnerOnline, setRunnerOnline] = useState(false);
  const [attentionSupported, setAttentionSupported] = useState(false);
  const [attentionResult, setAttentionResult] = useState<AttentionResult | null>(null);
  const [attentionSource, setAttentionSource] = useState<"recorded" | "live">("recorded");
  const [attentionBusy, setAttentionBusy] = useState(false);
  const [attentionError, setAttentionError] = useState("");
  const [gauntlet, setGauntlet] = useState<GauntletEvidence | null>(null);
  const selected = workloads.find(w => w.id === caseId)!;
  const baselineMemory = result?.baseline.peak_allocated_mb;
  const candidateMemory = result?.candidate.peak_allocated_mb;
  const memoryReduction = baselineMemory && candidateMemory !== null && candidateMemory !== undefined && baselineMemory > candidateMemory
    ? Math.round((1 - candidateMemory / baselineMemory) * 100)
    : null;
  const slowdown = result && result.baseline.latency_ms > 0 && result.candidate.latency_ms > result.baseline.latency_ms
    ? (result.candidate.latency_ms / result.baseline.latency_ms).toFixed(1)
    : null;

  useEffect(() => {
    fetch("/api/health", { cache: "no-store" }).then(response => response.json())
      .then(data => {
        if (Array.isArray(data.supported_cases)) setSupportedCases(data.supported_cases);
        setAttentionSupported(data.attention_supported === true);
        setRunnerOnline(data.runner_online === true);
        // On the hosted site, lead with proof from the saved local GPU run.
        // A fresh-run button remains available when the local runner is online.
        if (data.runner_online !== true) {
          fetch("/api/recorded-run?case_id=nearest", { cache: "no-store" })
            .then(response => response.ok ? response.json() : null)
            .then(saved => { if (saved) { setResult(saved); setResultSource("recorded"); } })
            .catch(() => {});
        }
      })
      .catch(() => {});
    fetch("/api/attention-evidence", { cache: "no-store" }).then(response => response.ok ? response.json() : null)
      .then(data => { if (data) { setAttentionResult(data); setAttentionSource("recorded"); } })
      .catch(() => {});
    fetch("/api/contract-gauntlet", { cache: "no-store" }).then(response => response.ok ? response.json() : null)
      .then(data => { if (data?.checks_total) setGauntlet(data); })
      .catch(() => {});
  }, []);

  async function run() {
    setBusy(true); setError(""); setResult(null);
    try {
      const response = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case_id: caseId, size, memory_limit_mb: memory, latency_limit_ms: latency }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Run failed");
      setResult(data);
      setResultSource("live");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Run failed");
    } finally { setBusy(false); }
  }

  async function loadRecorded() {
    setBusy(true); setError(""); setResult(null); setRecordedDecision(null); setDecisionError("");
    try {
      const response = await fetch(`/api/recorded-run?case_id=${encodeURIComponent(caseId)}`, { cache: "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Saved run unavailable");
      setResult(data);
      setResultSource("recorded");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Saved run unavailable");
    } finally { setBusy(false); }
  }

  async function askNemotron() {
    setDecisionBusy(true); setDecisionError(""); setRecordedDecision(null);
    try {
      const response = await fetch("/api/recorded-decision", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ case_id: result?.case_id }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Nemotron request failed");
      setRecordedDecision(data);
    } catch (cause) {
      setDecisionError(cause instanceof Error ? cause.message : "Nemotron request failed");
    } finally { setDecisionBusy(false); }
  }

  async function runAttention() {
    setAttentionBusy(true); setAttentionError("");
    try {
      const response = await fetch("/api/attention", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size: 1536, width: 64, memory_budget_mib: 8 }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Attention run failed");
      setAttentionResult(data);
      setAttentionSource("live");
    } catch (cause) {
      setAttentionError(cause instanceof Error ? cause.message : "Attention run failed");
    } finally { setAttentionBusy(false); }
  }

  return <main>
    <div className="ambient ambient-one"/><div className="ambient ambient-two"/>
    <header className="topbar shell">
      <div className="brand"><span className="brand-icon">H<span>.</span></span><span>HEADROOM</span></div>
      <nav aria-label="Main navigation"><a href="#workbench">Workbench</a><a href="#attention">Attention</a><a href="#gauntlet">Proof lab</a></nav>
      <div className="top-right"><span className="live-dot"/> GPU REPAIR LAB <span className="version">v0.1</span></div>
    </header>

    <section className="hero shell">
      <div className="eyebrow"><span className="eyebrow-line"/> BUILT FOR THE MOMENT YOUR GPU SAYS NO</div>
      <h1>Make it fit.<br/><em>Prove it works.</em></h1>
      <p>Headroom finds the memory bottleneck, proposes a bounded repair, and tests whether the outputs you actually need still survive.</p>
      <div className="hero-actions"><a className="hero-link" href="#workbench">Open the workbench <ArrowDownRight size={19}/></a><a className="hero-source" href="https://github.com/sujalrao061/headroom" target="_blank" rel="noreferrer">Browse the open source code ↗</a></div>
      <div className="hero-mark" aria-hidden="true"><span>H</span><i/></div>
    </section>

    <section id="workbench" className="workbench shell">
      <div className="section-heading"><div><div className="kicker">01 / THE WORKBENCH</div><h2>One memory pattern. Different promises.</h2></div><p>Choose a built-in workload and set its constraints. No code uploads, account, or personal data needed.</p></div>
      <div className="work-grid">
        <div className="panel setup-panel">
          <div className="panel-heading"><span className="panel-step">01</span><div><h3>Choose a workload</h3><p>Same allocation pattern. Different output contract.</p></div></div>
          <div className="workload-list">{workloads.filter(w => supportedCases.includes(w.id)).map(w => <button key={w.id} className={`workload ${caseId === w.id ? "selected" : ""}`} onClick={() => {setCaseId(w.id); setResult(null); setRecordedDecision(null); setDecisionError("");}} aria-pressed={caseId === w.id}>
            <span className="workload-top"><span className="workload-tag">{w.tag}</span><span className="radio"/></span>
            <strong>{w.title}</strong><small>{w.summary}</small><span className="contract"><span>OUTPUT CONTRACT</span><b>{w.formula}</b></span>
          </button>)}</div>
          <div className="control-group"><div className="control-title"><h4>Set the limits</h4><CircleHelp size={15}/></div>
            <label>Point count <span>{size.toLocaleString()} × {size.toLocaleString()}</span><input type="range" min="64" max="1536" step="64" value={size} onChange={e => setSize(Number(e.target.value))}/></label>
            <label>Memory budget <span>{memory} MB</span><input type="range" min="8" max="512" step="8" value={memory} onChange={e => setMemory(Number(e.target.value))}/></label>
            <label>Latency ceiling <span>{latency} ms</span><input type="range" min="50" max="3000" step="50" value={latency} onChange={e => setLatency(Number(e.target.value))}/></label>
          </div>
          <button className="run-button" onClick={run} disabled={busy || !runnerOnline}><Play size={16} fill="currentColor"/>{busy ? "Running measured validation…" : runnerOnline ? "Run analysis" : "Live GPU runner offline"}<ArrowRight size={17}/></button>
          <button className="recorded-button" onClick={loadRecorded} disabled={busy}>Explore saved GPU run <ArrowRight size={16}/></button>
          {error && <p className="error" role="alert">{error}</p>}
          <p className="setup-note"><ShieldCheck size={15}/> Fixed workloads only · local runner · no user code execution</p>
        </div>

        <div className="panel evidence-panel" id="evidence">
          <div className="panel-heading"><span className="panel-step">02</span><div><h3>Trace the pressure</h3><p>Source-linked evidence, then an independent decision.</p></div><span className="panel-state">{result ? resultSource === "recorded" ? "SAVED GPU RUN" : "NEW GPU RUN" : "AWAITING RUN"}</span></div>
          <div className="code-window"><div className="code-head"><span><i/><i/><i/></span><b>{selected.id === "nearest" ? "nearest.py" : selected.id === "radius" ? "radius_count.py" : "matrix.py"}</b><span>PYTORCH</span></div><pre>{(result?.source ?? `def ${selected.id === "nearest" ? "nearest" : selected.id === "radius" ? "radius_count" : "matrix"}(query, reference):\n    delta = query[:, None, :] - reference[None, :, :]\n    distance = torch.sqrt(torch.sum(delta * delta, dim=-1))\n    return ${selected.id === "nearest" ? "torch.argmin(distance, dim=1)" : selected.id === "radius" ? "torch.sum(distance <= 1.0, dim=1)" : "distance"}`).split("\n").map((line, i) => <span key={i} className={i === 1 ? "hot-line" : ""}><b>{String(i + 1).padStart(2, "0")}</b>{line}</span>)}</pre><div className="code-foot"><span className="pulse"/> FULL PAIRWISE ALLOCATION <span>·</span> O(N × M) MEMORY</div></div>
          <div className="memory-card"><div className="mini-title"><span>MEMORY PROFILE</span><span>{result ? "MEASURED PEAK ADDITIONAL ALLOCATION" : "RUN TO MEASURE"}</span></div>{result ? <div className="bars"><div><label>Baseline <b>{result.baseline.peak_allocated_mb ?? "—"} MB</b></label><div className="bar-track"><i style={{width: `${Math.max(2, Math.min(100, ((result.baseline.peak_allocated_mb ?? 0) / Math.max(result.baseline.peak_allocated_mb ?? 1, result.candidate.peak_allocated_mb ?? 1))*100))}%`}}/></div></div><div><label>Candidate <b>{result.candidate.peak_allocated_mb ?? "—"} MB</b></label><div className="bar-track"><i className="candidate-bar" style={{width: `${Math.max(2, Math.min(100, ((result.candidate.peak_allocated_mb ?? 0) / Math.max(result.baseline.peak_allocated_mb ?? 1, result.candidate.peak_allocated_mb ?? 1))*100))}%`}}/></div></div></div> : <div className="empty-chart"><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/><span/></div>}{memoryReduction !== null && result?.device.includes("NVIDIA") && <p className="memory-summary">{memoryReduction}% less peak additional allocation{slowdown ? ` · ${slowdown}× slower than baseline` : ""}. The validator checks whether this tradeoff fits the stated limits.</p>}</div>
          {result ? <div className={`decision ${result.accepted ? "accept" : "reject"}`}><div className="decision-icon">{result.accepted ? <Check size={22}/> : <X size={22}/>}</div><div><span>INDEPENDENT VALIDATOR</span><h3>{result.accepted ? "Candidate accepted" : "Candidate rejected"}</h3><p>{result.reasons[0]}</p></div></div> : <div className="awaiting"><div className="awaiting-icon">◎</div><span>Your measured result will appear here.</span></div>}
        </div>
      </div>
    </section>

    {result && <section className="results shell" aria-live="polite"><div className="section-heading"><div><div className="kicker">02 / THE DECISION · {resultSource === "recorded" ? "RECORDED GPU EVIDENCE" : "NEW GPU MEASUREMENT"}</div><h2>Show the work. Then trust the result.</h2>{resultSource === "recorded" && <p className="results-caveat">Captured {result.recorded_at ? new Date(result.recorded_at).toLocaleString() : "on a local GPU"}. The controls above do not change this saved run.</p>}</div><div className="downloads"><button className="download" onClick={() => saveReport(result)}><Download size={17}/> JSON report</button><button className="download" onClick={() => saveRegression(result)}><Download size={17}/> Regression test</button></div></div>
      <div className="result-grid"><div className="result-card"><span className="result-label">CONTRACT CHECKS</span><h3>{result.checks.filter(c => c.passed).length} / {result.checks.length} passed</h3><div className="checks">{result.checks.map(c => <div key={c.shape}><span className={c.passed ? "pass-icon" : "fail-icon"}>{c.passed ? <Check size={14}/> : <X size={14}/>}</span>{c.shape}<small>{c.passed ? "Matched" : `Expected [${c.expected_shape.join(", ")}], got [${c.actual_shape.join(", ")}]`}</small></div>)}</div></div>
      <div className="result-card"><span className="result-label">PERFORMANCE</span><h3>Measured, not estimated.</h3><div className="metric"><span>Baseline median</span><b>{result.baseline.latency_ms} ms</b></div><div className="metric"><span>Candidate median</span><b>{result.candidate.latency_ms} ms</b></div><div className="metric"><span>Peak total allocated</span><b>{result.candidate.peak_total_allocated_mb ?? "Unverified"} MB</b></div><div className="metric"><span>Peak reserved</span><b>{result.candidate.peak_reserved_mb ?? "Unverified"} MB</b></div><p className="metric-foot">{result.device} · {result.repetitions} timed runs · {result.warmups} warm-up · PyTorch {result.torch_version}</p></div>
      <div className="result-card"><span className="result-label">NEMOTRON VIA NEBIUS</span><h3>{recordedDecision ? "Live model response" : result.nemotron.status === "live" ? "Live model response" : result.nemotron.status === "error" ? "Model request failed" : resultSource === "recorded" ? "Ask about this saved run" : "Live call not configured"}</h3><p>{recordedDecision ? `Nemotron chose ${recordedDecision.strategy}. It ${recordedDecision.agrees_with_validator ? "agrees with" : "differs from"} the independent validator. The GPU measurements are recorded; the model choice was made now.` : result.nemotron.explanation}</p><div className="model-state"><span className={recordedDecision || result.nemotron.status === "live" ? "live-dot" : "off-dot"}/>{recordedDecision?.model ?? result.nemotron.model ?? (resultSource === "recorded" ? "Recorded GPU evidence only" : "Set NEBIUS_API_KEY on the runner")}</div>{resultSource === "recorded" && <><button className="recorded-button" onClick={askNemotron} disabled={decisionBusy}>{decisionBusy ? "Asking Nemotron…" : "Ask Nemotron now"}<ArrowRight size={16}/></button>{decisionError && <p className="error" role="alert">{decisionError}</p>}</>}</div></div>
      <div className="patch-card"><div><span className="result-label">REVIEWABLE REPAIR · {result.candidate_origin.toUpperCase()}</span><h3>{result.patch}</h3><p>Regression assertion: <code>{result.regression_test}</code></p></div><span className="patch-outcome">{result.accepted ? "VALIDATED ON THIS DEVICE" : "NOT ACCEPTED"}</span></div>
      <pre className="patch-diff">{result.patch_diff}</pre>
      <div className="library-compare"><div><span className="result-label">STANDARD LIBRARY COMPARISON</span><h3>What about torch.cdist?</h3><p>A measured alternative using PyTorch's existing distance operation. Tiling is not a new algorithm.</p></div><div className="library-metrics"><span><b>{result.library.peak_allocated_mb ?? "—"} MiB</b> peak allocated</span><span><b>{result.library.latency_ms} ms</b> median latency</span><span className={result.library.within_limits ? "frontier-pass" : "frontier-fail"}>{result.library.within_limits ? "MEETS THIS CONTRACT & BUDGET" : result.library.contract_passed ? "EXCEEDS THIS BUDGET" : "OUTPUT CHECK FAILED"}</span></div></div>
      {result.frontier && <div className="frontier"><div><span className="result-label">TESTED OPERATING RANGE</span><h3>Largest size tested within limits: {result.frontier.largest_tested_within_limits?.toLocaleString() ?? "none"} × {result.frontier.largest_tested_within_limits?.toLocaleString() ?? "none"}</h3><p>Only these sizes were measured. Untested sizes are unknown.</p></div><div className="frontier-rows">{result.frontier.tested.map(row => <div key={row.size}><b>{row.size} × {row.size}</b><span>{row.peak_allocated_mb ?? "—"} MiB</span><span>{row.latency_ms} ms</span><strong className={row.within_limits ? "frontier-pass" : "frontier-fail"}>{row.within_limits ? "WITHIN LIMITS" : "OUTSIDE LIMITS"}</strong></div>)}</div></div>}
      <p className="results-caveat">This run checks {result.checks.length} finite input cases, including tile-edge values. It does not establish correctness for every input or the largest feasible workload. No physical CUDA OOM was observed unless the report explicitly says so.</p>
    </section>}

    {(attentionResult || attentionSupported) && <section id="attention" className="attention-section shell" aria-live="polite"><div className="section-heading"><div><div className="kicker">03 / SECOND WORKLOAD FAMILY</div><h2>Attention: choose the measured winner.</h2></div><p>Full attention, Python tiling, and PyTorch SDPA face the same output checks and 8 MiB memory budget.</p></div>
      <div className="attention-panel"><div><span className="result-label">SINGLE-HEAD FLOAT32 ATTENTION</span><h3>Do not mistake low memory for the best repair.</h3><p>The hand-written tiled version can fit the budget while running much slower. Headroom measures the standard library before recommending either.</p>{attentionSupported ? <button className="run-button attention-run" onClick={runAttention} disabled={attentionBusy}><Play size={16} fill="currentColor"/>{attentionBusy ? "Measuring attention…" : "Run attention comparison"}<ArrowRight size={17}/></button> : <p className="attention-limit">Recorded local evidence is shown. Restart the updated GPU runner to enable a new run.</p>}{attentionError && <p className="error" role="alert">{attentionError}</p>}</div>
      {attentionResult && <div className="attention-outcome"><span className="result-label">{attentionSource === "recorded" ? "RECORDED LOCAL GPU RUN" : "NEW LOCAL GPU RUN"}</span><h3>Recommend {attentionResult.recommended === "pytorch_sdpa" ? "PyTorch SDPA" : attentionResult.recommended === "tiled_python" ? "Python tiling" : attentionResult.recommended === "keep_baseline" ? "keep the original" : "no verified option"}</h3><div className="metric"><span>Full matrix</span><b>{attentionResult.baseline.peak_allocated_mb ?? "—"} MiB · {attentionResult.baseline.latency_ms} ms</b></div><div className="metric"><span>Python tiling</span><b>{attentionResult.candidate.peak_allocated_mb ?? "—"} MiB · {attentionResult.candidate.latency_ms} ms</b></div><div className="metric"><span>PyTorch SDPA</span><b>{attentionResult.library.peak_allocated_mb ?? "—"} MiB · {attentionResult.library.latency_ms} ms</b></div><p>{attentionResult.checks.filter(c => c.passed && c.library_passed).length}/{attentionResult.checks.length} sampled output checks passed for both alternatives. {attentionResult.device}.</p><div className="model-state"><span className={attentionResult.nemotron.status === "live" ? "live-dot" : "off-dot"}/>{attentionResult.nemotron.status === "live" ? `${attentionSource === "recorded" ? "Recorded Nemotron decision" : "Nemotron"}: ${attentionResult.nemotron.strategy}; ${attentionResult.model_agrees ? "agrees with" : "differs from"} the measurement.` : attentionResult.nemotron.explanation}</div>{attentionSource === "recorded" && <p className="attention-limit">Saved local run · {attentionResult.recorded_at ? new Date(attentionResult.recorded_at).toLocaleString() : "date unavailable"} · {attentionResult.evidence_verified ? "Model decision verified against this GPU report" : "Model decision unverified"}</p>}<p className="attention-limit">{attentionResult.limitations}</p></div>}</div>
    </section>}
    {gauntlet && <section id="gauntlet" className="gauntlet-section shell"><div className="section-heading"><div><div className="kicker">04 / CORRECTNESS GAUNTLET</div><h2>We tried to break the repair.</h2></div><p>Same output contracts across uneven and asymmetric shapes, duplicate-heavy and tightly clustered inputs, and adversarial tile boundaries. This is a correctness stress run, not a speed claim.</p></div><div className="gauntlet-panel"><div className="gauntlet-score"><span className="result-label">RECORDED {gauntlet.device.includes("NVIDIA") ? "GPU" : "CPU"} VALIDATION</span><strong>{gauntlet.checks_passed}<small> / {gauntlet.checks_total}</small></strong><span className={gauntlet.passed ? "frontier-pass" : "frontier-fail"}>{gauntlet.passed ? "ALL CHECKS PASSED" : "FAILURES FOUND"}</span><p>{gauntlet.parameters.random_shapes} shapes × {gauntlet.parameters.seeds_per_shape} seeds × {gauntlet.parameters.tile_sizes.length} tile sizes, plus {gauntlet.parameters.stress_input_families?.length ?? 0} stress input families and cross-tile boundary cases.</p></div><div className="gauntlet-details"><div><span className={gauntlet.regression_sensitivity.nearest_that_ignores_later_reference_tiles_detected ? "pass-icon" : "fail-icon"}>{gauntlet.regression_sensitivity.nearest_that_ignores_later_reference_tiles_detected ? <Check size={14}/> : <X size={14}/>}</span><b>Injected bug caught</b><small>Ignoring later nearest-neighbor tiles</small></div><div><span className={gauntlet.regression_sensitivity.radius_that_excludes_exact_boundary_detected ? "pass-icon" : "fail-icon"}>{gauntlet.regression_sensitivity.radius_that_excludes_exact_boundary_detected ? <Check size={14}/> : <X size={14}/>}</span><b>Injected bug caught</b><small>Excluding points exactly at the radius</small></div><div className="gauntlet-meta">{gauntlet.device} · PyTorch {gauntlet.torch_version}<br/>Captured {new Date(gauntlet.captured_at_utc).toLocaleDateString()}</div><a className="download gauntlet-download" href="https://github.com/sujalrao061/headroom/blob/main/evidence/contract-gauntlet.json" target="_blank" rel="noreferrer"><Download size={16}/> Inspect every trial</a></div></div><p className="results-caveat">Finite reproducible checks increase confidence but cannot prove correctness for every possible tensor. The full trial report and command to regenerate it are in the open repository.</p></section>}
    <section id="method" className="method shell"><div className="section-heading"><div><div className="kicker">05 / THE METHOD</div><h2>Nothing passes on a good story alone.</h2></div></div><div className="method-grid"><article><span>01</span><h3>Locate the allocation</h3><p>Connect a growing tensor shape to the line that materializes it.</p></article><article><span>02</span><h3>Bound the repair</h3><p>Use a small, reviewable strategy set. Model output is never executed as code.</p></article><article><span>03</span><h3>Challenge the output</h3><p>Independent cases check shape, values, ties, memory, and latency.</p></article><article><span>04</span><h3>Keep the evidence</h3><p>Download the settings, measurements, decision, and regression assertion.</p></article></div></section>
    <footer className="footer shell"><div className="brand"><span className="brand-icon">H<span>.</span></span><span>HEADROOM</span></div><p>Build on evidence. Leave room to grow.</p><div className="footer-links"><a href="https://github.com/sujalrao061/headroom" target="_blank" rel="noreferrer">Source code ↗</a><a href="#workbench">Back to top ↑</a></div></footer>
  </main>;
}
