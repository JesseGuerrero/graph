# Project Instructions

- Commit often with descriptive messages
- Make small commits, not monolithic ones
- Write minimal code with full functionality
- Keep code simple and readable

# Storm Dependency

- The `storm/` folder is a local fork of the Stanford STORM paper repo, NOT a PyPI package
- Always use the local `storm/` source — never install from PyPI
- Install with `uv pip install -e storm/` to use the local source directly
- All modifications to STORM code go in `storm/knowledge_storm/` (the local fork)
- This project extends STORM for future work; eventually the local fork will be integrated directly into the project with proper citation to the original paper
