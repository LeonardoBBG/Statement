# Moltie Architecture — Quick Mental Model

## `schemas/` — meaning & control
Defines **what we are testing** and **how decisions look**.

- `QueryObject / AtomQuery`  
  → intent derived from WS → Y_JSON  
  → what legal proposition is being tested

- `Verdict`  
  → result of LLM verification  
  → support / contrast / harmful + anchors + confidence

- `NegativeExit`  
  → explicit “nothing to see here”  
  → stops compute on irrelevant / non-converging cases

Schemas never touch raw text.

---

## `corpus/` — perception & memory
Turns **raw appeals** into **searchable evidence units**.

Responsibilities:
1. Load full appeal texts (`loader.py`)
2. Split into stable paragraphs/chunks with IDs (`chunker.py`)
3. Surface *candidate* locations for a query (`index.py`)
4. Return evidence packs (paras + IDs) to the agent

Corpus does **cheap recall only**.
No reasoning. No legal judgment.

---

## `agent/` — reasoning & iteration
Controls the loop:
- retrieve candidates from corpus
- call LLM verifier
- refine query
- stop on convergence or NegativeExit

---

## `llm/` — semantic verification
Reads evidence packs and decides:
- relevance
- anchors
- support vs harm
- confidence

---

## Key principle
**Recall first (corpus), precision later (LLM).**  
Corpus answers *where to look*.  
Agent + LLM answer *what it means*.
