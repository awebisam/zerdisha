
# **PRD — Personal Exploration Engine (Typer TUI)**

## **1. Overview**

A terminal-based interactive tool built with **Typer** that serves as your *personal cognitive exoskeleton* for exploring topics, mapping conceptual connections, and refining your mental models.
The system integrates:

* **Conversational Agent (CA)** — your Socratic guide in metaphors
* **Pattern Detector (PD)** — maps session concepts to your Neo4j graph
* **Metacognitive Agent (MA)** — detects drift, stagnation, or metaphor lock-in and adjusts CA parameters

The TUI is **session-oriented** — each session is a contiguous exploration with the ability to save, review, and extend later.

---

## **2. Goals**

* Provide a **single command-line interface** for:

  * Running guided exploration sessions
  * Logging and visualizing connections
  * Triggering metacognitive adjustments
* Keep **Neo4j** as the *single source of truth* for:

  * Graph nodes & edges (concepts, metaphors, canonical links)
  * Vector embeddings (u-vectors & c-vectors)
* Eliminate manual file juggling — the tool writes directly to the DB
* Maintain **fast startup** — load only when needed, not on launch

---

## **3. Non-Goals**

* Multi-user support
* Web UI or collaborative features
* Over-optimized onboarding for general learners
* Building for public release at this stage

---

## **4. Core Components**

### **4.1 Conversational Agent (CA)**

* Powered by OpenAI/GPT or local LLM
* Loads **persona.md** at session start
* Always references:

  * Your `philosophy.json` for worldview
  * Your metaphor connections for analogical reasoning
* Can call tools for:

  * **Canonical Check** — fetch c-vector info from Neo4j
  * **Simulation Hook** — suggest & run Python/NumPy snippets
  * **Experimental TODOs** — append concrete testable ideas

---

### **4.2 Pattern Detector (PD)**

* Hooks into every CA exchange
* Extracts:

  * New concepts (nodes)
  * Relationships (edges) — metaphorical or canonical
* Writes directly to Neo4j:

  * Creates/updates u-vectors & c-vectors
  * Connects across domains where metaphors overlap

---

### **4.3 Metacognitive Agent (MA)**

* Runs asynchronously during/after sessions
* Flags:

  * Metaphor lock-in (same analogy persisting too long)
  * Topic drift
  * Underexplored concepts
* Can trigger **persona adjustments** mid-session
* Suggests “seed discovery” — unexplored but promising concepts

---

## **5. TUI User Flow**

### **5.1 Start Session**

```bash
peengine start "Topic Name"
```

* Loads persona
* Creates session entry in Neo4j
* Optional: preload relevant past nodes

---

### **5.2 Explore**

During session:

```
CA> What's your current model of X?
You> ...
CA> [Guides with metaphor, asks Socratic questions]
```

* PD extracts & updates graph silently
* MA evaluates in background

---

### **5.3 Commands Inside Session**

* `/map` — Show session’s explored concepts & connections
* `/gapcheck` — Show mismatches between u-vector & c-vector
* `/seed` — Inject a new exploration seed from MA
* `/end` — Save and close session

---

### **5.4 Review Past Sessions**

```bash
peengine review 2025-08-09
```

* Pulls session transcript
* Shows:

  * Graph changes from session
  * u→c vector accuracy delta
  * Metaphor connection growth

---

## **6. Data Flow**

1. User input → CA generates response (guided Qs/metaphors)
2. PD parses CA+user messages → updates Neo4j
3. MA monitors graph changes → can adjust CA in real-time
4. End of session → summary saved to Neo4j & optional local file

---

## **7. Tech Stack**

* **Typer** — CLI & TUI commands
* **Neo4j** — Graph + vector store
* **OpenAI API / Local LLM** — CA/MA reasoning
* **Pydantic** — Data modeling for messages & graph entities
* **Rich** — For TUI rendering of transcripts and maps

---

## **8. Falsifiable Metrics**

* **Session Depth** — avg. concept link length from start node
* **Concept Delta** — # of unique nodes added per session
* **Metaphor Diversity** — # of distinct metaphors used
* **u→c Accuracy** — similarity score improvement over time

---

## **9. Risks**

* CA hallucinations polluting graph — mitigated by canonical checks
* MA over-adjusting — mitigated by manual override
* Vector drift — mitigated by periodic re-embedding

---