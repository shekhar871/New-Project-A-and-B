"""
FastAPI entrypoint expected by the OpenEnv hackathon layout (`server/app.py` next to the package).

We keep one process-wide environment for the simplest HTTP smoke tests; swap in a session
factory if you need concurrent WebSocket classrooms later.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from agentguard_gym.environment import AgentGuardEnvironment
from agentguard_gym.models import CyberTaskType
from agentguard_gym.realtime import EventBus, RealTimeTrainer, TrainerConfig
from agentguard_gym.grandfinals.adversarial_loop import AdversarialSystem
from agentguard_gym.models import CyberAdversarialTaskType

app = FastAPI(title="AgentGuard-Gym", version="0.2.0")
_env = AgentGuardEnvironment()
_bus = EventBus()
_trainer = RealTimeTrainer(_bus)
_adv: Optional[AdversarialSystem] = None


def _adversarial_system() -> AdversarialSystem:
    global _adv
    if _adv is None:
        import json
        from pathlib import Path

        seeds = json.loads(Path("data/attack_corpus.json").read_text(encoding="utf-8"))
        _adv = AdversarialSystem.from_seed_attacks(seeds)
    return _adv


class ResetBody(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0)
    episode_id: Optional[str] = None
    task: Optional[CyberTaskType] = None


class StepBody(BaseModel):
    action: Dict[str, Any]


@app.post("/reset")
def http_reset(body: ResetBody) -> Dict[str, Any]:
    obs = _env.reset(seed=body.seed, episode_id=body.episode_id, task=body.task)
    return {"observation": obs.model_dump(mode="json"), "reward": None, "done": False}


@app.post("/step")
def http_step(body: StepBody) -> Dict[str, Any]:
    result = _env.step(body.action)
    return {
        "observation": result.observation.model_dump(mode="json"),
        "reward": result.reward.model_dump(mode="json"),
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def http_state() -> Dict[str, Any]:
    return _env.state().model_dump(mode="json")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/")
def root_redirect() -> Response:
    # Keep root simple: UI is optional and shouldn't surprise OpenEnv validators.
    return Response(status_code=307, headers={"Location": "/ui"})


@app.get("/ui", response_class=HTMLResponse)
def ui() -> str:
    # Single-file dashboard (no build step). Uses SSE from /events.
    # Inspired by the user-provided reference layout (sidebar + KPI cards + chart + activity feed).
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AgentGuard-Gym — Dashboard</title>
  <style>
    :root {
      --bg: #f4f6fb;
      --panel: #ffffff;
      --muted: #6c768f;
      --text: #101828;
      --line: #e6eaf2;
      --shadow: 0 10px 30px rgba(16,24,40,0.08);

      --sidebar: #0f172a;
      --sidebar2: #111c36;
      --brand: #2f55ff;
      --ok: #12b76a;
      --bad: #f04438;
      --warn: #f79009;
      --chip: #f2f4f7;
    }

    * { box-sizing: border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

    .layout { display:flex; min-height: 100vh; }
    .sidebar {
      width: 92px; background: linear-gradient(180deg, var(--sidebar), var(--sidebar2));
      padding: 14px 10px; display:flex; flex-direction:column; gap:10px; align-items:center;
    }
    .brand {
      width: 54px; height: 54px; border-radius: 16px; background: rgba(255,255,255,0.10);
      display:flex; align-items:center; justify-content:center; color:#fff; font-weight:800;
      letter-spacing: 0.5px; border: 1px solid rgba(255,255,255,0.12);
    }
    .nav { margin-top: 12px; display:flex; flex-direction:column; gap:10px; width:100%; align-items:center; }
    .nav button {
      width: 56px; height: 44px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.06); color:#d5dbef; cursor:pointer; font-size:12px;
    }
    .nav button.active { background: rgba(47,85,255,0.20); border-color: rgba(47,85,255,0.55); color:#fff; }
    .sidebar .spacer { flex: 1; }

    .content { flex:1; padding: 18px 18px 28px; }
    header {
      display:flex; align-items:center; justify-content:space-between;
      background: transparent; margin-bottom: 14px;
    }
    .hello { display:flex; gap:14px; align-items:center; }
    .hello .title { font-size:20px; font-weight:800; }
    .hello .sub { font-size:13px; color:var(--muted); margin-top:3px; }
    .search { display:flex; gap:10px; align-items:center; }
    .search input {
      width: 420px; max-width: 52vw;
      border: 1px solid var(--line); border-radius: 14px;
      padding: 12px 14px; background: #fff;
      box-shadow: 0 1px 0 rgba(16,24,40,0.02);
    }
    .pill { padding: 6px 10px; border-radius: 999px; font-size: 12px; background: var(--chip); border: 1px solid var(--line); color: var(--muted); }
    .pill.ok { color: var(--ok); }
    .pill.bad { color: var(--bad); }
    .pill.warn { color: var(--warn); }
    a { color: var(--brand); text-decoration:none; }
    a:hover { text-decoration: underline; }

    .grid { display:grid; grid-template-columns: 1.3fr 0.8fr; gap: 14px; align-items:start; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--shadow); }
    .card .hd { padding: 14px 16px; border-bottom: 1px solid var(--line); display:flex; align-items:center; justify-content:space-between; }
    .card .hd .h { font-weight: 800; }
    .card .bd { padding: 14px 16px; }

    .kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
    .kpi { border: 1px solid var(--line); border-radius: 16px; padding: 12px 12px; background: #fff; }
    .kpi .l { font-size: 12px; color: var(--muted); }
    .kpi .v { margin-top: 6px; font-size: 22px; font-weight: 900; }

    .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    select, button, input.small {
      border: 1px solid var(--line); border-radius: 14px;
      padding: 10px 12px; background: #fff; color: var(--text); font-size: 14px;
    }
    button.primary { background: var(--brand); border-color: var(--brand); color:#fff; }
    button.ghost { background: #fff; }
    button.danger { background: #fff; border-color: rgba(240,68,56,0.45); color: var(--bad); }

    .tabs { display:flex; gap:8px; }
    .tab { padding: 8px 10px; border-radius: 999px; border: 1px solid var(--line); background: #fff; cursor:pointer; font-size: 13px; color: var(--muted); }
    .tab.active { background: rgba(47,85,255,0.10); border-color: rgba(47,85,255,0.35); color: var(--text); font-weight: 700; }

    .split { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .chartWrap { border: 1px solid var(--line); border-radius: 16px; padding: 10px; }
    canvas { width: 100%; height: 220px; display:block; }

    .feed { height: 420px; overflow:auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 12px; background: #0b1020; color: #e8ecff; border-radius: 16px; padding: 12px; border: 1px solid #1f2a4b; }
    .feed .muted { color: #a4b0d2; }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">AG</div>
      <div class="nav">
        <button id="nav_live" class="active" title="Live">Live</button>
        <button id="nav_train" title="Training">Train</button>
        <button id="nav_eval" title="Eval">Eval</button>
      </div>
      <div class="spacer"></div>
      <a class="pill" href="/docs" target="_blank" rel="noreferrer" style="background:rgba(255,255,255,0.08); border-color:rgba(255,255,255,0.12); color:#fff;">API</a>
      <div style="height:10px;"></div>
    </aside>
    <section class="content">
      <header>
        <div class="hello">
          <div>
            <div class="title">AgentGuard Dashboard</div>
            <div class="sub">Real-time environment + training signals (SSE stream from <code>/events</code>).</div>
          </div>
          <span id="conn" class="pill warn">connecting</span>
        </div>
        <div class="search">
          <input id="search" placeholder="Search events (e.g. SSRF, FN, TP, entropy)"/>
          <span class="pill">T4-safe: G=4</span>
        </div>
      </header>

      <div class="grid">
        <div class="card">
          <div class="hd">
            <div class="h">Live control</div>
            <div class="tabs">
              <button class="tab active" id="tab_sim">Simulator</button>
              <button class="tab" id="tab_adv">Adversarial</button>
            </div>
          </div>
          <div class="bd">
            <div class="kpis">
              <div class="kpi"><div class="l">Simulator running</div><div class="v" id="running">—</div></div>
              <div class="kpi"><div class="l">Episodes</div><div class="v" id="episodes">0</div></div>
              <div class="kpi"><div class="l">Mean score (last 100)</div><div class="v" id="mean100">0.00</div></div>
              <div class="kpi"><div class="l">Last score</div><div class="v" id="lastscore">0.00</div></div>
            </div>

            <div style="height:12px;"></div>
            <div class="controls" id="controls_sim">
              <select id="policy">
                <option value="heuristic" selected>Heuristic policy</option>
                <option value="random">Random policy</option>
              </select>
              <button class="primary" id="start">Start</button>
              <button class="danger" id="stop">Stop</button>
              <select id="task">
                <option value="">Any task</option>
                <option value="prompt_injection">prompt_injection</option>
                <option value="tool_misuse_ssrf">tool_misuse_ssrf</option>
                <option value="memory_poisoning_privilege">memory_poisoning_privilege</option>
              </select>
              <input class="small" id="seed" placeholder="seed (optional)" style="width:140px"/>
              <button class="ghost" id="runone">Run one episode</button>
            </div>

            <div class="controls" id="controls_adv" style="display:none;">
              <select id="adv_task">
                <option value="">Any adversarial task</option>
                <option value="prompt_injection">prompt_injection</option>
                <option value="tool_misuse_ssrf">tool_misuse_ssrf</option>
                <option value="memory_poisoning">memory_poisoning</option>
              </select>
              <input class="small" id="adv_step" placeholder="step_idx (default 0)" style="width:180px"/>
              <button class="primary" id="run_adv">Run adversarial episode</button>
              <button class="ghost" id="refresh_adv">Refresh curriculum</button>
              <span class="pill" id="adv_status">difficulty —</span>
              <span class="pill" id="elo_status">ELO —</span>
            </div>

            <div style="height:12px;"></div>
            <div class="split">
              <div class="chartWrap">
                <div class="muted" style="font-size:12px; margin:4px 4px 8px;">Reward (recent)</div>
                <canvas id="chart_reward" width="800" height="260"></canvas>
              </div>
              <div class="chartWrap">
                <div class="muted" style="font-size:12px; margin:4px 4px 8px;">Entropy / stability (recent)</div>
                <canvas id="chart_entropy" width="800" height="260"></canvas>
              </div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="hd">
            <div class="h">Activity feed</div>
            <div class="controls">
              <button class="ghost" id="clear">Clear</button>
            </div>
          </div>
          <div class="bd">
            <div id="log" class="feed" aria-label="event log"></div>
          </div>
        </div>
      </div>
    </section>
  </div>
<script>
  const $ = (id) => document.getElementById(id);
  const log = $("log");
  const rewardSeries = [];
  const entropySeries = [];

  function append(line, cls="") {
    const q = ($("search").value || "").trim().toLowerCase();
    if (q && !line.toLowerCase().includes(q)) return;
    const div = document.createElement("div");
    div.textContent = line;
    if (cls) div.className = cls;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
  }

  function drawSeries(canvas, series, {min=0, max=1, color="#2f55ff"}={}) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0,0,W,H);
    ctx.strokeStyle = "rgba(16,24,40,0.10)";
    ctx.lineWidth = 1;
    for (let i=1;i<5;i++){
      const y = (H/5)*i;
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
    }
    if (!series.length) return;
    const n = series.length;
    const pad = 10;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i=0;i<n;i++){
      const x = pad + (W-2*pad) * (i/(Math.max(1,n-1)));
      const v = Math.max(min, Math.min(max, series[i]));
      const y = pad + (H-2*pad) * (1 - ((v-min)/(max-min || 1)));
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  }

  function pushSeries(arr, v, cap=80){
    arr.push(v);
    while(arr.length>cap) arr.shift();
  }

  async function api(path, body=null) {
    const res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? {"content-type":"application/json"} : {},
      body: body ? JSON.stringify(body) : null
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function refreshStatus() {
    const s = await api("/trainer/status");
    $("running").textContent = s.running ? "yes" : "no";
    $("episodes").textContent = String(s.episodes);
    $("mean100").textContent = (s.mean_reward_100 ?? 0).toFixed(2);
    $("lastscore").textContent = (s.last_episode_score ?? 0).toFixed(2);
  }

  async function refreshAdversarial() {
    const r = await api("/adversarial/curriculum");
    const c = r.curriculum || {};
    const e = r.elo || {};
    $("adv_status").textContent = `difficulty ${(c.difficulty ?? 0).toFixed(2)} | win10 ${(c.win_rate_10 ?? 0).toFixed(2)}`;
    $("elo_status").textContent = `ELO A ${(e.attacker_elo ?? 1200).toFixed(0)} / D ${(e.defender_elo ?? 1200).toFixed(0)}`;
  }

  $("start").onclick = async () => {
    const policy = $("policy").value;
    await api("/trainer/start", {policy});
    await refreshStatus();
  };
  $("stop").onclick = async () => {
    await api("/trainer/stop", {});
    await refreshStatus();
  };
  $("runone").onclick = async () => {
    const policy = $("policy").value;
    const task = $("task").value || null;
    const seedRaw = $("seed").value.trim();
    const seed = seedRaw === "" ? null : Number(seedRaw);
    const out = await api("/episode/run", {policy, task, seed});
    append(`[episode] id=${out.episode_id} policy=${out.policy} score=${out.score.toFixed(2)}`);
    for (const step of out.trace) {
      append(`  step action=${step.action.defense} r=${Number(step.reward.value).toFixed(2)} done=${step.done} outcome=${step.reward.outcome}`);
    }
  };
  $("clear").onclick = () => { log.textContent = ""; };

  $("run_adv").onclick = async () => {
    const task = $("adv_task").value || null;
    const stepRaw = ($("adv_step").value || "").trim();
    const step_idx = stepRaw === "" ? 0 : Number(stepRaw);
    const out = await api("/adversarial/episode", {task, step_idx});
    append(`[adversarial] ep=${out.episode_id.slice(0,8)} task=${out.task} decision=${out.defender_decision} outcome=${out.outcome} rD=${Number(out.reward_defender).toFixed(2)} rA=${Number(out.reward_attacker).toFixed(2)}`);
    await refreshAdversarial();
  };
  $("refresh_adv").onclick = async () => { await refreshAdversarial(); };

  // tabs
  function setTab(which){
    const sim = which === "sim";
    $("tab_sim").classList.toggle("active", sim);
    $("tab_adv").classList.toggle("active", !sim);
    $("controls_sim").style.display = sim ? "flex" : "none";
    $("controls_adv").style.display = sim ? "none" : "flex";
  }
  $("tab_sim").onclick = () => setTab("sim");
  $("tab_adv").onclick = () => { setTab("adv"); refreshAdversarial().catch(()=>{}); };

  const es = new EventSource("/events");
  es.onopen = () => { $("conn").textContent = "connected"; $("conn").className="pill ok"; };
  es.onerror = () => { $("conn").textContent = "disconnected"; $("conn").className="pill bad"; };
  es.addEventListener("episode.step", (e) => {
    const d = JSON.parse(e.data);
    append(`[step] ep=${d.episode_id.slice(0,8)} task=${d.task} step=${d.step} action=${d.action.defense} r=${Number(d.reward).toFixed(2)} outcome=${d.outcome}${d.partial_credit ? " partial" : ""}`);
    pushSeries(rewardSeries, Number(d.reward) || 0);
    drawSeries($("chart_reward"), rewardSeries, {min:0, max:1, color:"#2f55ff"});
  });
  es.addEventListener("episode.ended", (e) => {
    const d = JSON.parse(e.data);
    append(`[end] ep=${d.episode_id.slice(0,8)} task=${d.task} score=${Number(d.score).toFixed(2)} mean100=${Number(d.mean_score_100).toFixed(2)}`);
    refreshStatus().catch(()=>{});
  });
  es.addEventListener("trainer.started", (e) => {
    const d = JSON.parse(e.data);
    append(`[trainer] started policy=${d.policy}`);
    refreshStatus().catch(()=>{});
  });
  es.addEventListener("trainer.stopped", () => {
    append(`[trainer] stopped`);
    refreshStatus().catch(()=>{});
  });
  es.addEventListener("error", (e) => {
    const d = JSON.parse(e.data);
    append(`[error] ${d.message}`);
  });

  // If training writes entropy values to SSE later, we'll pick them up via [entropy] logs.
  // For now: render empty charts.
  drawSeries($("chart_reward"), rewardSeries, {min:0, max:1, color:"#2f55ff"});
  drawSeries($("chart_entropy"), entropySeries, {min:0, max:5, color:"#12b76a"});

  refreshStatus().catch(()=>{});
</script>
</body>
</html>"""


class TrainerStartBody(BaseModel):
    policy: str = Field(default="heuristic", pattern="^(heuristic|random)$")
    steps_per_episode: int = Field(default=12, ge=1, le=50)
    episodes_per_minute: int = Field(default=30, ge=1, le=600)
    seed: int = Field(default=42, ge=0)


@app.post("/trainer/start")
def trainer_start(body: TrainerStartBody) -> Dict[str, Any]:
    status = _trainer.start(
        TrainerConfig(
            policy=body.policy,
            steps_per_episode=body.steps_per_episode,
            episodes_per_minute=body.episodes_per_minute,
            seed=body.seed,
        )
    )
    return status.__dict__


@app.post("/trainer/stop")
def trainer_stop() -> Dict[str, Any]:
    return _trainer.stop().__dict__


@app.get("/trainer/status")
def trainer_status() -> Dict[str, Any]:
    return _trainer.status().__dict__


class EpisodeRunBody(BaseModel):
    policy: str = Field(default="heuristic", pattern="^(heuristic|random)$")
    task: Optional[CyberTaskType] = None
    seed: Optional[int] = Field(default=None, ge=0)
    steps_limit: int = Field(default=12, ge=1, le=50)


@app.post("/episode/run")
def episode_run(body: EpisodeRunBody) -> Dict[str, Any]:
    return _trainer.run_single_episode(task=body.task, seed=body.seed, policy=body.policy, steps_limit=body.steps_limit)


@app.get("/events")
def events() -> StreamingResponse:
    return StreamingResponse(_bus.stream(), media_type="text/event-stream")


class AdversarialEpisodeBody(BaseModel):
    task: Optional[CyberAdversarialTaskType] = None
    step_idx: int = Field(default=0, ge=0, le=1000)


@app.post("/adversarial/episode")
def adversarial_episode(body: AdversarialEpisodeBody) -> Dict[str, Any]:
    sys = _adversarial_system()
    res = sys.run_episode(task=body.task, step_idx=body.step_idx)
    _bus.emit(
        "episode.step",
        {
            "episode_id": res.episode_id,
            "task": res.task.value,
            "step": body.step_idx,
            "action": {"defense": res.defender_decision},
            "reward": res.reward_defender,
            "outcome": res.outcome,
            "partial_credit": False,
            "done": True,
        },
    )
    _bus.emit(
        "episode.ended",
        {
            "episode_id": res.episode_id,
            "task": res.task.value,
            "score": res.reward_defender,
            "mean_score_100": sys.curriculum.status().get("win_rate_10") or 0.0,
        },
    )
    return res.model_dump(mode="json")


@app.get("/adversarial/curriculum")
def adversarial_curriculum() -> Dict[str, Any]:
    sys = _adversarial_system()
    return {"curriculum": sys.curriculum.status(), "elo": sys.elo.model_dump(mode="json")}


@app.get("/adversarial/corpus_stats")
def adversarial_corpus_stats() -> Dict[str, Any]:
    sys = _adversarial_system()
    return {
        "entries": len(sys.corpus.entries),
        "mean_novelty": (sum(e.novelty_score for e in sys.corpus.entries) / max(1, len(sys.corpus.entries))),
        "by_task": {
            t.value: sum(1 for e in sys.corpus.entries if e.task.value == t.value) for t in CyberAdversarialTaskType
        },
    }


def main() -> None:
    """CLI entry used by `uv run server` / OpenEnv validators (binds PORT from env, default 8000)."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
