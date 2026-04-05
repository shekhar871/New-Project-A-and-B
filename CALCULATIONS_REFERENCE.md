# Calculations & algorithms — AgentGuard-Gym + AML-DefenseGym

Single audit page for **both** hackathon environments. Each project stays **independent** (separate repos / Spaces). The same file is copied to each repo root as `CALCULATIONS_REFERENCE.md` so a one-repo HF clone still has the full math; edit the parent copy under `SCALAR/` and re-copy if you change formulas.

**Source of truth (code):**


| Project        | Core numerics                                                  | Normalization                    | Episode / HTTP  |
| -------------- | -------------------------------------------------------------- | -------------------------------- | --------------- |
| AgentGuard-Gym | `agentguard_gym/graders.py`, `agentguard_gym/reward_math.py`   | `agentguard_gym/environment.py`  | `server/app.py` |
| AML-DefenseGym | `aml_defense_gym/graders.py`, `aml_defense_gym/reward_math.py` | `aml_defense_gym/environment.py` | `server/app.py` |


**Inference episode score (both `inference.py`):**

\text{episodescore} = \frac{1}{T} \sum_{t=1}^{T} r_t
\quad\text{where}\quad r_t = \texttt{reward.value} \in [0,1]

Success flag (baseline scripts): \text{episodescore} \ge 0.65.

---

## Shared primitives

### Clamp

\text{clamp}_{[0,1]}(x) = \min(1,\max(0,x))

**Code:** `aml_defense_gym/reward_math.py` → `clamp01` (AML). AgentGuard uses the same logic inside `minmax_normalize` for the final clip.

### Min–max to [0,1]

For raw utility U with conservative bounds [U_{\min}, U_{\max}]:

\text{norm}(U) = \text{clamp}*{[0,1]}\left(\frac{U - U*{\min}}{U_{\max} - U_{\min}}\right)

If U_{\max} \le U_{\min}, the code falls back to \text{clamp}_{[0,1]}(U).

**Code:** `reward_math.py` → `minmax_normalize` in **both** projects.

---

## AgentGuard-Gym (cyber)

### Step 1 — Outcome & base weight

Let `outcome` \in \text{tp},\text{tn},\text{fp},\text{fn}. Base utility (before timing / extras):

U_{\text{base}} = w_{\text{outcome}} + f_{\text{action}}

**Defaults** (`RewardConfig` in `agentguard_gym/graders.py`):


| Symbol                           | Value   |
| -------------------------------- | ------- |
| w_{\text{tp}}                    | `1.0`   |
| w_{\text{tn}}                    | `0.15`  |
| w_{\text{fp}}                    | `-0.6`  |
| w_{\text{fn}}                    | `-2.5`  |
| f_{\text{action}} (`action_fee`) | `-0.02` |


### Step 2 — Task-specific multipliers / surcharges

- **SSRF**, true positive with partial audit (`AUDIT_TOOL_CHAIN` on internal URL):  
U \leftarrow U \times \texttt{partialauditfraction} (default `**0.82`**).
- **Memory**, true positive with coarse `BLOCK` (partial):  
U \leftarrow U \times 0.9.
- **False negative surcharges** (extra penalty when the attack is let through):
  - Prompt injection: `+ fn_extra_prompt` = `**-3.0`**
  - SSRF: `+ fn_extra_ssrf` = `**-4.0`**
  - Memory: `+ fn_extra_memory` = `**-3.5`**

(Here “+” means adding a negative number.)

### Step 3 — MTTD / MTTR-style potential (bounded)

Let `detected_step` / `remediated_step` be step indices maintained by the environment when a threat is handled correctly.

\phi_{\text{time}} =
\frac{\alpha}{1 + d_{\text{det}}}
+
\frac{\beta}{1 + \max(0, d_{\text{rem}} - d_{\text{det}})}

This is **added** to raw utility (not subtracted): larger delay ⇒ **smaller** bonus ⇒ **lower** total utility, so faster detection/remediation is rewarded.

**Defaults:** \alpha = `mttd_scale` `**0.15`**, \beta = `mttr_scale` `**0.1`**.

**Code:** `reward_math.py` → `mttd_mttr_step_potential`; called from each `grade`_* in `graders.py`.

### Step 4 — Raw utility & normalization bounds

U_{\text{raw}} = U_{\text{base}} + \phi_{\text{time}} + \text{surcharges/multiples as above}

**Bounds for min–max** (`cyber_minmax_bounds`):

U_{\min} = w_{\text{fn}} + f_{\text{action}} + \min(\texttt{fnextraprompt}, \texttt{fnextrassrf}, \texttt{fnextramemory})

U_{\max} = w_{\text{tp}} + \texttt{mttdscale} + \texttt{mttrscale} + f_{\text{action}}

**Displayed reward:** r = \text{norm}(U_{\text{raw}}) → stored in `AgentGuardReward.value`.

### Special cases

- **Invalid action (Pydantic error):** U_{\text{raw}} = -2.0, same (U_{\min}, U_{\max}) as above.
- **Episode already finished:** `reward.value = 0.0`.

### Task ↔ difficulty (for reviewers)


| Task enum                    | Difficulty |
| ---------------------------- | ---------- |
| `prompt_injection`           | easy       |
| `tool_misuse_ssrf`           | medium     |
| `memory_poisoning_privilege` | hard       |


---

## AML-DefenseGym

### A) Sanctions screening

**Raw utility** from disposition vs hidden `true_match` (`grade_sanctions`):

- True match + **escalate:** U = C_{\text{TP}} + f  
- True match + **clear:** U = C_{\text{FN}} + f  
- True match + **request_info:** U = 0.35C_{\text{FN}} + f (partial path; `partial_credit=True`)
- False match + **clear:** U = C_{\text{TN}} + f  
- False match + **escalate:** U = C_{\text{FP}} + f  
- False match + **request_info:** U = 0.7C_{\text{TN}} + f (`partial_credit=True`)

**Defaults** (`AMLRewardConfig`):


| Symbol        | Parameter    | Value   |
| ------------- | ------------ | ------- |
| C_{\text{TP}} | `illicit_tp` | `5.0`   |
| C_{\text{FN}} | `illicit_fn` | `-25.0` |
| C_{\text{TN}} | `legit_tn`   | `0.5`   |
| C_{\text{FP}} | `legit_fp`   | `-2.0`  |
| f             | `action_fee` | `-0.01` |


**Normalization:**

U_{\min} = C_{\text{FN}} + f,\quad U_{\max} = C_{\text{TP}} + f

r = \text{norm}(U)

**Missing disposition:** U = -2.0 with **sanctions** bounds (same U_{\min}, U_{\max}).

### B) EDD review

**Raw score** (`grade_edd`), no match → **−5**; missing keys → **−2.5**; shallow text → **−1**; complete → **+2**.

**Normalization interval:** [U_{\min}, U_{\max}] = [-5, 2].

r = \text{norm}(U)

**Validation / task mismatch** uses the **EDD** interval for mapping fixed penalties (−3.0, −1.8) — see `environment.py`.

### C) Transaction monitoring (batch)

Let L_i \in 0,1 be labels (`is_laundering`), s_i \in [0,1] agent scores.  
Let `recall@k` = fraction of illicit rows captured in top‑k by s (descending).  
Let `fp_pressure` = fraction of **benign** rows inside that top‑k.

k = \min\bigl(n,\ \max(1,\ |\{i: L_i=1\}|)\bigr)

So the alert budget can cover **all** illicit rows in the batch (capped only by batch size `n`). A perfect ranker can achieve `recall@k = 1` whenever all positives fit in the batch.

U_{\text{raw}} = \text{clamp}_{[0,1]}\bigl(0.65 \cdot \text{recall@k} + 0.35 \cdot (1 - \text{fppressure})\bigr)

Here **no extra min–max**: r = U_{\text{raw}} already in [0,1].

**Code:** `aml_defense_gym/graders.py` → `grade_transaction_batch`.

### Task ↔ difficulty


| Task enum                | Difficulty |
| ------------------------ | ---------- |
| `sanctions_screening`    | easy       |
| `edd_review`             | medium     |
| `transaction_monitoring` | hard       |


---

## Validation & baselines (what we ran locally)

- **OpenEnv CLI (must pass for Round 1):** from each project directory, with Python **3.11** and `openenv` on `PATH` (e.g. `~/Library/Python/3.11/bin` after `pip install --user openenv-core`):
  ```bash
  openenv validate --verbose .
  ```
  Latest check: **`[OK] agentguard-gym: Ready for multi-mode deployment`** and **`[OK] aml-defense-gym: Ready for multi-mode deployment`** (docker, `openenv_serve`, `uv run`, `python_module` all **YES**).
- **Reproducible scores without an LLM:** `uv sync --extra dev` then `PYTHONPATH=. python scripts/offline_baseline.py` → writes `**baseline_scores.json`** (oracle policy; upper-ish bound on the graders).

---

## Cross-check: `Verification and Improvement of Calculations - Google Docs.pdf`

The PDF in the same **`SCALAR/`** folder audits this document against RL and SOC/AML practice. Here is how it lines up with the **current code**:

| PDF section | Verdict | Notes |
|-------------|---------|--------|
| **§5.1 MTTR “inversion”** | **Does not apply** | The paper models the timing term as **subtracted**. In code, `mttd_mttr_step_potential` is **added** to utility (`reward_math.py`). Shorter detection/remediation delays ⇒ **larger** additive bonus ⇒ **higher** utility. |
| **§5.2 min–max fallback** | **Edge case** | If `best <= worst`, we `clamp01(raw)`. For negative raw utilities that collapses many distinct failures toward `0` — unlikely with fixed `RewardConfig` bounds, but worth monitoring if bounds are ever data-driven. |
| **§5.3 partial multiplier on negatives** | **Low risk today** | Partial multipliers apply only on **TP** paths after outcome weights are applied; TP branch uses positive `w_tp`. Order of operations is in `graders.py`. |
| **§5.4 AML top‑k cap** | **Was valid; fixed** | Old `k = min(5, \|illicit\|+1)` could make max recall &lt; 1 for dense positives. **Now** `k = min(n, max(1, \|illicit\|))` in `grade_transaction_batch`. |
| **§5.5 Ng et al. potential shaping** | **Fair critique** | The timing term is an **informal** bounded bonus, not the strict \(F(s')-F(s)\) potential-difference that preserves optimal policies. We cite Ng et al. as **inspiration**, not a claim of full policy invariance. |

---

## References (high level)

- Ng, Harada & Russell — potential-based shaping (time term is a loose, bounded analogue).
- Elkan — cost-sensitive classification / asymmetric misclassification costs (AML sanctions).
- Industry TM narratives — recall vs analyst workload (blend weights in transaction monitoring).

For exact line-by-line behavior, always prefer the linked Python files above.