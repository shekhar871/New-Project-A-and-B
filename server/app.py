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


def _start_metrics_tailer() -> None:
    """
    Track 2: Surface real GRPO training signals in the live dashboard.
    If `runs/training_metrics.jsonl` exists (from a Colab/HF-Jobs run), we tail it
    and re-broadcast entries as SSE events so judges can see a live curve.
    """

    import json
    import threading
    import time
    from pathlib import Path

    path = Path("runs/training_metrics.jsonl")

    def _tail() -> None:
        last = 0
        while True:
            try:
                if not path.is_file():
                    time.sleep(1.0)
                    continue
                with path.open("r", encoding="utf-8") as f:
                    f.seek(last)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        _bus.emit(
                            "trainer.metrics",
                            {
                                "step": row.get("step"),
                                "win_rate": row.get("win_rate"),
                                "fp_rate": row.get("fp_rate"),
                                "entropy": row.get("entropy"),
                                "loss": row.get("loss"),
                            },
                        )
                    last = f.tell()
            except Exception:
                time.sleep(1.0)
            time.sleep(0.5)

    threading.Thread(target=_tail, daemon=True).start()


@app.on_event("startup")
def _on_startup() -> None:
    _start_metrics_tailer()


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
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AgentGuard-Gym — Live</title>
  <style>
    :root { --bg:#0b1020; --panel:#121a33; --muted:#9aa4c0; --text:#e8ecff; --ok:#39d98a; --bad:#ff5c7a; --warn:#ffd166; --line:#22305f; }
    body { margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
    header { padding:18px 18px 10px; border-bottom:1px solid var(--line); display:flex; gap:12px; align-items:center; justify-content:space-between; }
    .title { font-weight:700; letter-spacing:0.2px; }
    .sub { color:var(--muted); font-size:13px; margin-top:4px; }
    main { padding:18px; display:grid; grid-template-columns: 360px 1fr; gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    button, select, input { background:#0f1731; color:var(--text); border:1px solid var(--line); padding:10px 10px; border-radius:10px; font-size:14px; }
    button { cursor:pointer; }
    button.primary { background:#2a4bff; border-color:#2a4bff; }
    button.danger { background:#401324; border-color:#6a1e3a; }
    .kpi { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px;}
    .k { padding:10px; background:#0f1731; border:1px solid var(--line); border-radius:10px; }
    .k .l { color:var(--muted); font-size:12px; }
    .k .v { font-size:20px; font-weight:700; margin-top:6px; }
    .log { height:540px; overflow:auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size:12px; background:#070b17; border:1px solid var(--line); border-radius:12px; padding:12px; }
    .pill { padding:4px 8px; border-radius:999px; font-size:12px; border:1px solid var(--line); background:#0f1731; color:var(--muted); }
    .pill.ok { color:var(--ok); border-color:rgba(57,217,138,.35); }
    .pill.bad { color:var(--bad); border-color:rgba(255,92,122,.35); }
    .pill.warn { color:var(--warn); border-color:rgba(255,209,102,.35); }
    .sep { height:1px; background:var(--line); margin:12px 0; }
    a { color:#9db2ff; text-decoration:none; }
    a:hover { text-decoration:underline; }
  </style>
</head>
<body>
  <div style="background:#FAEEDA;border-bottom:1px solid #EF9F27;padding:7px 16px;font-size:13px;color:#633806;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', monospace">
    ⚡ Environment simulation (<b>heuristic/random</b> policy) in this dashboard · Real GRPO training →
    <code>uv run python train_adversarial.py</code>
  </div>
  <header>
    <div>
      <div class="title">AgentGuard-Gym — Live dashboard</div>
      <div class="sub">See episodes/training in real time. Backed by the same `AgentGuardEnvironment` + graders.</div>
    </div>
    <div class="row">
      <span id="conn" class="pill warn">connecting</span>
      <a class="pill" href="/docs" target="_blank" rel="noreferrer">API docs</a>
    </div>
  </header>
  <main>
    <section class="card">
      <div class="row">
        <select id="policy">
          <option value="heuristic" selected>Heuristic policy</option>
          <option value="random">Random policy</option>
        </select>
        <button class="primary" id="start">Start trainer</button>
        <button class="danger" id="stop">Stop</button>
      </div>
      <div class="kpi">
        <div class="k"><div class="l">Running</div><div class="v" id="running">—</div></div>
        <div class="k"><div class="l">Episodes</div><div class="v" id="episodes">0</div></div>
        <div class="k"><div class="l">Mean score (last 100)</div><div class="v" id="mean100">0.00</div></div>
        <div class="k"><div class="l">Last score</div><div class="v" id="lastscore">0.00</div></div>
      </div>
      <div class="sep"></div>
      <div class="row">
        <select id="task">
          <option value="">Any task</option>
          <option value="prompt_injection">prompt_injection</option>
          <option value="tool_misuse_ssrf">tool_misuse_ssrf</option>
          <option value="memory_poisoning_privilege">memory_poisoning_privilege</option>
        </select>
        <input id="seed" placeholder="seed (optional)" style="width:140px"/>
        <button id="runone">Run one episode</button>
      </div>
      <div class="sub" style="margin-top:10px;">
        Tip: run a single episode to inspect step-by-step rewards/outcomes in the log.
      </div>
      <div class="sep"></div>
      <div class="title" style="font-size:14px;margin-bottom:8px;">Learning curve (streamed)</div>
      <div class="sub" style="margin-bottom:10px;">
        Green = win_rate, Red = fp_rate from <code>runs/training_metrics.jsonl</code> if present.
      </div>
      <canvas id="chart" width="330" height="140" style="width:100%;background:#070b17;border:1px solid var(--line);border-radius:12px;"></canvas>
    </section>
    <section class="card">
      <div class="row" style="justify-content:space-between;">
        <div class="title" style="font-size:14px;">Live event log</div>
        <div class="row">
          <span class="pill">SSE: <code>/events</code></span>
          <button id="clear">Clear</button>
        </div>
      </div>
      <div id="log" class="log" aria-label="event log"></div>
    </section>
  </main>
<script>
  const $ = (id) => document.getElementById(id);
  const log = $("log");
  const metrics = [];
  function redrawChart() {
    const c = $("chart");
    if (!c) return;
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.clearRect(0,0,W,H);
    ctx.fillStyle = "#070b17";
    ctx.fillRect(0,0,W,H);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    for (let i=0;i<5;i++){
      const y = (H-20) - i*((H-30)/4);
      ctx.beginPath(); ctx.moveTo(10,y); ctx.lineTo(W-10,y); ctx.stroke();
    }
    const pts = metrics.slice(-200);
    if (pts.length < 2) return;
    function plot(key, color){
      ctx.strokeStyle = color;
      ctx.beginPath();
      for (let i=0;i<pts.length;i++){
        const x = 10 + (i*(W-20)/(pts.length-1));
        const v = Math.max(0, Math.min(1, Number(pts[i][key] ?? 0)));
        const y = (H-20) - v*(H-30);
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
    }
    plot("win_rate", "#39d98a");
    plot("fp_rate", "#ff5c7a");
  }
  function append(line, cls="") {
    const div = document.createElement("div");
    div.textContent = line;
    if (cls) div.className = cls;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
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

  const es = new EventSource("/events");
  es.onopen = () => { $("conn").textContent = "connected"; $("conn").className="pill ok"; };
  es.onerror = () => { $("conn").textContent = "disconnected"; $("conn").className="pill bad"; };
  es.addEventListener("episode.step", (e) => {
    const d = JSON.parse(e.data);
    append(`[step] ep=${d.episode_id.slice(0,8)} task=${d.task} step=${d.step} action=${d.action.defense} r=${Number(d.reward).toFixed(2)} outcome=${d.outcome}${d.partial_credit ? " partial" : ""}`);
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
  es.addEventListener("trainer.metrics", (e) => {
    const d = JSON.parse(e.data);
    metrics.push(d);
    redrawChart();
  });
  es.addEventListener("trainer.stopped", () => {
    append(`[trainer] stopped`);
    refreshStatus().catch(()=>{});
  });
  es.addEventListener("error", (e) => {
    const d = JSON.parse(e.data);
    append(`[error] ${d.message}`);
  });

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

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
