---
project: alphatafl
type: agent-orientation
---

# AGENTS.md — Project orientation for LLM agents

This file is the first stop for any LLM agent working on the **alphatafl** codebase.
Its only job is to point you to the canonical knowledge base and the rules you must follow.

## Knowledge base

The project's persistent, LLM-maintained wiki lives in **`alphatafl/`** (an Obsidian vault — see `alphatafl/.obsidian/`).

Key entry points:

- `alphatafl/index.md` — index of every page in the wiki.
- `alphatafl/log.md` — chronological log of wiki updates.
- `alphatafl/docs/` — the actual documentation, organised as:
  - `docs/architecture/` — system design
  - `docs/engine/` — C++ game engine
  - `docs/model/` — neural network
  - `docs/training/` — self-play, training loop, inference server
  - `docs/api/` — Python/C++ bindings, CLI, GUI, interface contract
  - `docs/history/` — versioned changelogs (v0, v1, v1 benchmarks, v1 phase 5, …)
- `alphatafl/WIKI.md` — **the wiki schema**. Read this before editing any wiki page. It defines directory structure, conventions, and workflows.

## Hard rules

1. **Do not create a root-level `wiki/` directory.** A previous version of the schema sent wiki files there, which produced a stale duplicate that no one read. The canonical wiki is `alphatafl/`. If you find references to a root-level `wiki/` in any old document, treat them as historical and do not act on them.
2. **Current-state pages must reflect the current code only.** No "removed in vX" blocks, no "replaces old X" parentheticals, no v0 bug mentions in `docs/api/`, `docs/architecture/`, `docs/engine/`, `docs/model/`, or `docs/training/`. Historical changes belong in `docs/history/` and `log.md`.
3. **Every page change updates `alphatafl/index.md` and `alphatafl/log.md`.** See the Workflows section in `alphatafl/WIKI.md`.

## When you start a session

1. Read `alphatafl/WIKI.md` for the full schema, conventions, and workflows.
2. Skim `alphatafl/index.md` to see what pages already exist.
3. Skim `alphatafl/log.md` for recent context (last few entries).
4. If a current-state page disagrees with the code, the page is wrong — fix it. Cite `path:line` from the source file.
5. If a history page disagrees with older log entries, the history page is wrong (the log is the chronological record of what actually happened).
