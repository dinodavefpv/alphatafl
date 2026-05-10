# Wiki Schema & Conventions

This wiki serves as the persistent knowledge base for the AlphaTafl project. It is maintained by an LLM agent using the `llm-wiki` skill.

## Directory Structure

- `wiki/index.md`: The main index of all pages in the wiki.
- `wiki/log.md`: A chronological log of all wiki updates.
- `wiki/docs/`: Detailed documentation pages.
    - `wiki/docs/history/`: Version history and changelog (v0, v1, etc.).
    - `wiki/docs/architecture/`: High-level system design.
    - `wiki/docs/engine/`: C++ game engine details.
    - `wiki/docs/model/`: Neural network architecture.
    - `wiki/docs/training/`: Training and self-play processes.
    - `wiki/docs/api/`: API and bindings documentation.

> **Agent Note:** All new wiki pages and directories must be created **inside `wiki/`**. Do not place wiki files at the project root or in folders outside of `wiki/`.

## Conventions

- **Markdown**: All pages are written in GitHub-flavored Markdown.
- **Interlinking**: Pages should be heavily interlinked using relative paths (e.g., `[Engine](./engine/README.md)`).
- **Citations**: Wiki pages should cite source files and specific lines when possible.
- **Consistency**: The LLM agent is responsible for maintaining consistency across all pages.
- **Current-state only**: Pages in `api/`, `architecture/`, `engine/`, `model/`, and `training/` must reflect only the current implementation. Never reference legacy code, removed features, deprecated functions, or past bugs in these sections — no "removed in vX" blocks, no "replaces old X" parentheticals, no v0 bug mentions. Historical changes, removals, and design decisions belong exclusively in `wiki/docs/history/` and `wiki/log.md`.
- **Frontmatter**: Each page should have a basic YAML frontmatter for metadata.

## Workflows

1. **Ingest**: Analyze a source file or component and create/update relevant wiki pages.
2. **Update Index**: Every time a page is created or modified, update `wiki/index.md`.
3. **Log Entry**: Every action should be recorded in `wiki/log.md`.
