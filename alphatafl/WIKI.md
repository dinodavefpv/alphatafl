# Wiki Schema & Conventions

This wiki serves as the persistent knowledge base for the AlphaTafl project. It is maintained by an LLM agent using the `llm-wiki` skill.

## Directory Structure

- `alphatafl/index.md`: The main index of all pages in the wiki.
- `alphatafl/log.md`: A chronological log of all wiki updates.
- `alphatafl/docs/`: Detailed documentation pages.
    - `alphatafl/docs/history/`: Version history and changelog (v0, v1, etc.).
    - `alphatafl/docs/architecture/`: High-level system design.
    - `alphatafl/docs/engine/`: C++ game engine details.
    - `alphatafl/docs/model/`: Neural network architecture.
    - `alphatafl/docs/training/`: Training and self-play processes.
    - `alphatafl/docs/api/`: API and bindings documentation.

> **Agent Note:** The wiki is fully consolidated and tracked in git under the `alphatafl/` folder. All new wiki pages and directories must be created **inside `alphatafl/docs/`** (or in `alphatafl/` for index and log updates). Do not place wiki files outside of `alphatafl/`.

## Conventions

- **Markdown**: All pages are written in GitHub-flavored Markdown.
- **Interlinking**: Pages should be heavily interlinked using relative paths (e.g., `[Engine](./engine/README.md)`).
- **Citations**: Wiki pages should cite source files and specific lines when possible.
- **Consistency**: The LLM agent is responsible for maintaining consistency across all pages.
- **Current-state only**: Pages in `api/`, `architecture/`, `engine/`, `model/`, and `training/` must reflect only the current implementation. Never reference legacy code, removed features, deprecated functions, or past bugs in these sections — no "removed in vX" blocks, no "replaces old X" parentheticals, no v0 bug mentions. Historical changes, removals, and design decisions belong exclusively in `alphatafl/docs/history/` and `alphatafl/log.md`.
- **Frontmatter**: Each page should have a basic YAML frontmatter for metadata.

## Workflows

1. **Ingest**: Analyze a source file or component and create/update relevant wiki pages.
2. **Update Index**: Every time a page is created or modified, update `alphatafl/index.md`.
3. **Log Entry**: Every action should be recorded in `alphatafl/log.md`.
