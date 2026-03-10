# OllamaRunner — Design Document

## Vision
A self-contained, lightweight, Windows-focused local LLM runner with a web UI. Zero manual setup — auto-downloads llama.cpp CUDA binaries, scans for GGUF models, and provides full control over inference parameters with intelligent resource management.

## Architecture

### Principles
- **Modular**: Each subsystem is an independent module with clean interfaces. Swap backends, add tools, extend UI without touching core.
- **Performance-first**: CUDA-optimized inference, WebSocket streaming (no buffering), minimal frontend (no framework bloat), async throughout.
- **Extensible**: Plugin-style tool system, composable system prompt, per-model config overrides.
- **Good UX**: Smart defaults, resource-aware guardrails, real-time feedback, dark theme.

### System Diagram
```
┌─────────────────────────────────────────┐
│            Browser (Web UI)             │
│   Vanilla HTML/CSS/JS + WebSocket       │
│   Chat │ Models │ Settings │ Canvas     │
└──────────────┬──────────────────────────┘
               │ HTTP + WebSocket
┌──────────────┴──────────────────────────┐
│         FastAPI Backend (async)         │
│                                         │
│  ┌───────────┐ ┌───────────┐           │
│  │  Router    │ │ WebSocket │           │
│  │  Layer     │ │ Manager   │           │
│  └─────┬─────┘ └─────┬─────┘           │
│        │              │                  │
│  ┌─────┴──────────────┴─────┐           │
│  │      Service Layer       │           │
│  │  ┌─────────┐ ┌────────┐ │           │
│  │  │ Model   │ │Session │ │           │
│  │  │ Manager │ │Manager │ │           │
│  │  ├─────────┤ ├────────┤ │           │
│  │  │ System  │ │Config  │ │           │
│  │  │ Monitor │ │Manager │ │           │
│  │  ├─────────┤ ├────────┤ │           │
│  │  │ Tool    │ │Prompt  │ │           │
│  │  │ Engine  │ │Composer│ │           │
│  │  └─────────┘ └────────┘ │           │
│  └──────────────────────────┘           │
└──────────────┬──────────────────────────┘
               │ HTTP (OpenAI-compatible)
┌──────────────┴──────────────────────────┐
│     llama-server (CUDA, managed)        │
│     Auto-downloaded, per-model process  │
└─────────────────────────────────────────┘
```

### Module Responsibilities

**ModelManager** — Scans model directory for GGUF files, reads metadata (param count, quant, max context), manages llama-server process lifecycle (start/stop/restart), calculates optimal GPU layer offloading.

**SystemMonitor** — Real-time VRAM/RAM/CPU monitoring via pynvml + psutil. Detects whether NVIDIA GPU is driving a display. Provides pre-load feasibility checks and runtime health monitoring.

**SessionManager** — SQLite-backed chat history. Create/list/delete/rename sessions. Stores messages with role, content, timestamp, token counts.

**ConfigManager** — JSON-based persistent config. App-level settings + per-model overrides. Human-editable files.

**ToolEngine** — Pluggable tool system. Each tool is a module with: schema (for LLM function calling), execute method, toggle state. MVP tools: web search, file system.

**PromptComposer** — Assembles system prompt from: system-prompt.md + identity.md + user.md + memory.md + active tool schemas. Clean separation of concerns.

**WebSocket Manager** — Proxies SSE stream from llama-server to browser WebSocket. Token-by-token, zero buffering.

## Data Layout
```
C:\Users\omar-\.ollamarunner\
├── config.json              # App config
├── identity.md              # LLM persona
├── user.md                  # User profile
├── memory.md                # LLM self-maintained memory
├── system-prompt.md         # Base system prompt
├── models.json              # Model registry (path → config)
├── sessions/
│   └── sessions.db          # SQLite
└── bin/
    └── llama-server.exe     # Auto-downloaded
```

### Model Registry (models.json)
```json
{
  "models": [
    {
      "id": "abc123",
      "name": "Qwen3 8B Q4",
      "path": "C:\\Users\\omar-\\.lmstudio\\models\\qwen3\\Qwen3-8B-Q4_K_M.gguf",
      "gpu_layers": 33,
      "context_default": 4096,
      "context_recommended": 8192,
      "context_max": 32768,
      "temperature": 0.7,
      "top_p": 0.9
    }
  ]
}
```

### Model Directory
Default scan path: `C:\Users\omar-\.lmstudio\models`
Configurable in config.json. Recursive scan for *.gguf files.

## Resource Guardrails

### VRAM/RAM Limits
- VRAM: up to 95% if GPU has no display attached, 90% otherwise
- RAM: 85% of total (leave OS headroom)
- Detected via nvidia-smi / pynvml at runtime

### Pre-load Check
Before starting llama-server:
1. Calculate VRAM needed = gpu_layers × layer_size
2. Calculate RAM needed = remaining_layers × layer_size + context_memory
3. Compare against currently available (not total) resources
4. Block load if exceeded, show explanation + suggestions

### Runtime Monitoring
- Poll every 2 seconds while model is running
- UI shows live CPU/GPU/RAM bars
- Warning at 85% usage
- Context fill indicator with warning before OOM

### Context Length Control
- Default: conservative starting point (e.g., 4096)
- Recommended: auto-calculated based on available memory after model load
- Max: read from GGUF metadata
- UI slider with markers at all three levels
- Hard-capped at what system can handle

## Web UI

### Layout
```
┌──────────┬──────────────────────────────┐
│ Sidebar  │  Chat Area                   │
│          │                              │
│ Sessions │  Messages (streaming)        │
│ ───────  │  Markdown rendered           │
│ > Chat 1 │                              │
│   Chat 2 │                              │
│          │                              │
│ ───────  │                              │
│ Model:   │                              │
│ [picker] │  ┌────────────────────────┐  │
│ ───────  │  │ Input          [Send]  │  │
│ CPU ██░░ │  └────────────────────────┘  │
│ GPU ██░░ │  [Web Search: ON] [Files: OFF]│
│ RAM ███░ │                              │
└──────────┴──────────────────────────────┘
```

### Design Principles
- Dark theme, clean, minimal
- No framework — vanilla HTML/CSS/JS
- Streaming: tokens appear as generated via WebSocket
- Responsive resource bars update live
- Model picker: modal with GGUF tree browser + metadata + config sliders
- Settings: in-browser editors for system prompt, identity, user, memory files

## Streaming Flow
```
llama-server ---> SSE (server-sent events)
     │
FastAPI backend ---> reads SSE stream
     │
WebSocket ---> pushes each token to browser
     │
Browser JS ---> appends to chat, renders markdown incrementally
```

## Tool System

### Interface
Each tool implements:
- `name: str`
- `description: str`
- `schema: dict` (JSON schema for function calling)
- `async execute(params) -> result`
- `enabled: bool` (toggleable per session)

### MVP Tools
1. **Web Search** — DuckDuckGo (no API key), returns top results as context
2. **File System** — Read/write in configurable sandboxed directory

### Prompt Integration
If model supports native function calling → use that format.
Otherwise → structured text prompt with tool descriptions.

## System Prompt Composition
Assembled at chat time in order:
1. `system-prompt.md` — base rules
2. `identity.md` — persona
3. `user.md` — user context
4. `memory.md` — persistent facts
5. Active tool schemas

## Memory Management
- After each conversation, LLM prompted to extract key facts
- Appended to memory.md
- Capped at ~500 lines, LLM summarizes when approaching limit
- Human-editable

## Implementation Tiers

### Tier 1 — MVP Core
1. Project scaffolding + dependency management
2. Auto-download llama-server CUDA binary
3. Model scanner + GGUF metadata reader
4. System resource detection (VRAM/RAM/CPU)
5. Model manager (lifecycle, GPU layers, guardrails)
6. Chat backend (WebSocket streaming)
7. Session persistence (SQLite)
8. Web UI (chat, model picker, resources, sessions)
9. Config persistence
10. System prompt / identity / user / memory files

### Tier 2 — Tools
11. Web search tool
12. File system tool
13. In-UI editing for prompt/identity/user/memory

### Tier 3 — Enhanced
14. Canvas (Monaco editor, code execution)
15. Vision/media for capable models

### Tier 4 — API
16. Expose as authenticated API server

## Tech Stack
- Python 3.11+
- FastAPI (async, WebSocket, lightweight)
- SQLite (session storage)
- pynvml + psutil (system monitoring)
- Vanilla HTML/CSS/JS (no build step)
- llama-server (CUDA, auto-managed)
