# Project Instructions

- Commit often with descriptive messages
- Make small commits, not monolithic ones
- Write minimal code with full functionality
- Keep code simple and readable

# Storm Dependency

- The `src/` folder is a local fork of the Stanford STORM paper repo — we are building on top of it for a new paper and will cite the original
- `knowledge-storm` is NOT in `pyproject.toml` and NOT installed in `.venv` — it is loaded purely via `sys.path` at runtime
- `storm_runner.py` registers `src/` as `knowledge_storm` via `sys.modules` so all imports work
- All modifications go in `src/` and take effect immediately
- **BANNED: Never install knowledge-storm via pip, uv pip, uv pip install -e, or by adding it to pyproject.toml in any form**

# Package Manager

- This project uses `uv`, not conda
- Run the server with: `cd web && uv run uvicorn app:app --host 127.0.0.1 --port 8005`
