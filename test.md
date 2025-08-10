
# Action-Oriented Dialectic (AOD) — the daily loop

**Cadence:** 60 minutes total (20 + 20 + 20).
**Definition of Done:** one tangible artifact + one canonical check passed.

## 0) Setup (once)

* Pick one topic per week (e.g., “Kalman filter intuition”, “DC motor back-EMF”).
* Create a session in your TUI: `pe start "<topic>"` (your Typer tool will wire CA/PD/MA + Neo4j).

---

## 1) The Storm — 20 min (embrace the bias)

Goal: generate hypotheses & metaphors fast, on-record.

Prompts you run (or hotkeys in TUI):

* `/seed` to surface edges you usually miss.
* “List three *competing* metaphors for X. What would each *get wrong*?”
* “What would make this idea false in the physical world?”

**Log fields (auto or quick form):**

* `Hypotheses[]` (short)
* `Metaphors[]`
* `OpenQuestions[]`

**Guardrail:** no web/code here.

---

## 2) The Forge — 20 min (apply pressure)

Goal: falsify your best hypothesis.

Commands:

* `/gapcheck` → PD compares your u-vector to c-vector (Neo4j) and returns **gaps**.
* `/redteam` → switch persona to “Gunda” for a 10-minute cross-examination.
* “Give me *two canonical counterexamples* that break my metaphor.”

**Must produce:**

* `TopClaim` (one sentence)
* `Falsifiers[2–3]` (specific, feasible)
* `CanonRefs[2]` (textbook/lecture/standard you’ll map to)
* `FailureModes[2]` (where this likely collapses)

**Guardrail:** if you can’t state a falsifier, you’re not ready—go back to Storm.

---

## 3) The Artifact — 20 min (force reality)

Pick *one* and build it **now**:

* **Physics:** a NumPy/Matplotlib sim, or a worked derivation with a **numerical example**.
* **Robotics:** a testable script that toggles a real signal (e.g., PWM ramp with measured response), plus a **failure log** if it doesn’t work.

Commands:

* `/artifact "name"` → TUI scaffolds file + checklist
* `/todo experimental` → MA suggests the smallest measurable test

**Pass/Fail (non-negotiable):**

* Artifact runs / compiles / computes *or* you record the failure with error + next step.
* **Canonical parse test (10 min add-on if needed):**

  * Watch a 10–15 min lecture chunk on the same topic.
  * Write **5 bullets** mapping to your c-vector + do **1 worked example** from it.
  * If you *can’t* parse it, mark **No-Closure** → topic stays alive tomorrow.

---

# Templates (paste into your TUI as forms)

**Storm.md**

```
Topic: …
Hypotheses:
- …
Metaphors:
- …
OpenQuestions:
- …
```

**Forge.md**

```
TopClaim: …
Falsifiers:
- …
CanonRefs:
- [source,title,section]
FailureModes:
- …
```

**Artifact.md**

```
Objective: …
Method (steps):
Results (numbers / plots / logs):
Debug/Failures:
- …
Canon-Parse (5 bullets):
- …
Decision: Closure Y/N
Next Step (if N): …
```

---

# Metrics you actually track (lightweight, in Neo4j or CSV)

Per session:

* `closure` (Y/N)
* `u→c_gap_delta` (↓ is good)
* `artifact_type` (sim/derivation/code) & `artifact_pass` (Y/N)
* `metaphor_diversity` (# used)
* `time_spent` (always \~60)

Weekly:

* `closures/started` (≥ 3/5 good)
* `externalization_count` (# artifacts shared with future-you: gist/notebook)

**Kill switches:**

* Two consecutive **No-Closure** on same topic → pause topic, do 30-min pure **canon-first** session next day.
* Three artifacts in a row that “work” but you can’t pass the **canonical parse test** → you’re in prompt-driven illusion; shift a full day to derivations only.

---

# Anti–echo-chamber guardrails

* **Two-source rule:** Forge must cite **two** independent canonical sources before Artifact.
* **Red-team slot:** 10 minutes of Gunda every session (scheduled, not optional).
* **Constraint budget:** 1 “fancy tool” per artifact (e.g., you can use a library, *but* you must also hand-compute a small case).

---

# Rhythm

* **Daily:** one 60-min loop.
* **Weekly review (30 min, Friday):** scan graph diffs, pick next week’s topic, archive or revive open threads.
* **Monthly “capstone”:** one bigger artifact that integrates 2–3 topics (e.g., sensor fusion sim that uses your Kalman + control learnings).

---

# When to go public

Keep it private lab until you have **4 artifacts** that each pass:

* Canonical parse test
* A second brain (future-you) can run it cold in <10 min with notes
  Then open-source the **artifacts only**, not the raw sessions.

---