# Obsidian Assistant (Bev)

Local Obsidian assistant plugin + lightweight Python server. Includes the Bev character sprite sheet.

## Repository Layout
- `plugin/` Obsidian plugin files (copy to your vault).
- `server/` Local assistant server and helpers.

## Plugin Install
1. Copy `plugin/` into your vault:
   - `YourVault/.obsidian/plugins/obsidian-assistant/`
2. Restart Obsidian and enable **Obsidian Assistant** in Community Plugins.

The Bev sprite sheet is included at:
`plugin/assets/bev-poses.png`

## Server Setup
1. `cd server`
2. `cp config.example.json config.json`
3. Edit `config.json` to point to your vault and papers.
4. Start the server:
   - `python3 server.py`

Optional: start llama.cpp automatically with:
```
export LLAMA_MODEL="/path/to/model.gguf"
./bin/start.sh
```

## Notes
- The plugin expects the server at `http://127.0.0.1:8000`.
- The assistant uses only local files unless web search is enabled.
