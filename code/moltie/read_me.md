# Moltie Architecture — Quick Mental Model (v2)

## `schemas/` — meaning, contracts, stop-conditions
Defines **what we are testing** and **what outputs must look like**.

- `Y_Spec`  
  → rules/indicators extracted from WS (loose at first, tighten later)

- `QueryObject / AtomQuery`  
  → runtime intent derived from Y_Spec  
  → proposition + indicators + (later) expansions/synonyms

- `Verdict`  
  → the only “decision object” the LLM is allowed to emit  
  → relevant? support/contrast/harmful + anchors (para_id + verbatim quote) + confidence/score

- `NegativeExit`  
  → explicit “nothing here / don’t waste compute” outcome  
  → plateau, exhausted, no-anchors, mismatch, budget

Schemas don’t parse PDFs, don’t retrieve text, don’t run loops.

---

## `corpus/` — perception & evidence preparation
Turns **raw appeals** into **stable, addressable evidence units**.

Responsibilities:
1. Load documents + metadata (`loader.py`)
2. Split into stable paragraphs/chunks with IDs (`chunker.py`)
3. Provide cheap recall (keyword/token/embeddings later) (`index.py`)
4. Output *evidence packs*:
   - `doc_id`
   - `paras: [{para_id, text}, ...]`
   - retrieval metadata (method/score)

Corpus = recall. No legal reasoning.

---

## `llm/` — the barrister (single-shot reasoning)
LLM is a callable component that:
- reads an AtomQuery + evidence pack
- returns STRICT JSON `Verdict`
- must anchor every claim to **verbatim quotes** from provided paras
- does **not** loop, does **not** choose docs, does **not** decide when to stop

LLM = precision on provided evidence.

---

## `agent/` — the solicitor (orchestration & iteration)
Controls the end-to-end interrogation loop:
- `retrieve.py` → ask corpus for best paras for a given AtomQuery
- `loop.py` → iterate retrieve → verify → evaluate objective function
- `refine.py` → adjust query signal if weak (later: expansions, synonyms, negative cues)
- stop conditions:
  - accept when thresholds met (score/confidence/anchors)
  - NegativeExit when diverging/plateau/exhausted

Agent = state + strategy + stopping.

---

## `run.py` — wiring
Binds everything:
- Y_Spec → QueryObject
- corpus → agent
- agent → llm
- outputs: Verdicts + NegativeExits + trace/audit logs

---

## Key principles
1) **Recall first (corpus), precision later (LLM).**  
2) **Evidence-first:** no anchors ⇒ no “relevant=true”.  
3) **Agent owns the loop; LLM answers one question.**  
4) **NegativeExit is a feature, not a failure.**
