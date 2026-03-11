# Performance Optimization & UX Improvements

**Date:** 2026-03-11
**Status:** Draft

## Problem Statement

OllamaRunner has functional model loading, chat, and project management, but:
- The 27B model runs at 0.5 t/s (CPU-bottlenecked), making thinking models unusable
- llama-server launch flags are not optimized for the hardware
- No way to control thinking mode — models burn tokens on reasoning when not needed
- Frontend renders every token synchronously, causing jank during fast streaming
- UI has several rough edges: no feedback during operations, no keyboard shortcuts, resource monitor competes with streaming

## Hardware Constraints

- RTX 2070 Max-Q: 8GB VRAM (7.9GB usable, iGPU drives display)
- 32GB RAM, 6 physical / 12 logical cores
- Primary models: Qwen3.5-9B Q4_K_M (5.4GB, 32 layers) — sweet spot for this GPU
- Secondary: Qwen3.5-4B Q6_K (3.3GB) for fast iteration, 27B for occasional use

## Design

### Part A: llama-server Performance Tuning

**Changes to `model_manager.py` command construction:**

1. **KV cache quantization** — add `-ctk q8_0 -ctv q4_0`
   - Saves ~75% VRAM on KV cache (f16 → q4_0 for values, q8_0 for keys)
   - For 9B at 16K context: KV drops from ~128MB to ~40MB
   - Imperceptible quality impact per llama.cpp benchmarks

2. **Batch size alignment** — change `-b 512` to `-b 2048`
   - Default batch is 2048 but we hardcoded 512
   - Larger logical batch = faster prompt processing
   - Keep `-ub 512` (physical ubatch stays limited by VRAM)

3. **Thread priority** — add `--prio 2`
   - High priority for generation threads
   - Reduces scheduling latency on Windows

4. **Thread tuning** — add `-tb` (threads-batch) = physical core count
   - Use all physical cores for prompt processing (batch)
   - Generation threads stay at physical - 1 (current behavior, correct)

5. **Single slot** — add `-np 1`
   - Single user app, no need for parallel slots
   - Reduces memory overhead and scheduling complexity

6. **Remove `--no-mmap`** — we already use `--mlock`
   - `--no-mmap` forces read() instead of mmap, slower initial load
   - `--mlock` already pins pages in RAM, preventing swapping
   - Net effect: faster model loading, same runtime performance

7. **Skip warmup** — add `--no-warmup` (if supported by current build)
   - Skips KV cache warmup on load, reduces time-to-ready

8. **Continuous batching** — explicitly enable `-cb`
   - Should be default in server mode but explicit is safer

### Part B: Smart Defaults & Auto-Configuration

**New method `_compute_optimal_config()` in `model_manager.py`:**

Given a model's file size, layer count, and embedding dimension, plus current VRAM/RAM:

1. **Auto GPU layers**: Calculate max layers that fit after reserving VRAM for KV cache at target context
2. **Auto context**: If all layers fit on GPU, maximize context with remaining VRAM. Cap at model's native max.
3. **KV cache budget formula**:
   ```
   kv_bytes = embedding * context * 2 (K+V) * bytes_per_element
   q8_0: bytes_per_element ≈ 1.0625 (for K)
   q4_0: bytes_per_element ≈ 0.5625 (for V)
   average: ~0.8 bytes per element per KV pair
   ```
4. **Expose as "Recommended" in UI** — user can override but defaults are optimal

**Stored in ModelConfig as computed fields (not persisted, recalculated on load).**

### Part C: Thinking Mode Control

**Backend changes (`chat.py`):**

1. Add `thinking` field to the WebSocket message protocol
   - Client sends `{"content": "...", "thinking": true/false}`
   - Backend includes `"thinking": true/false` in the llama-server request body (Qwen3.5 supports this natively via `enable_thinking` parameter)

2. For models that don't support native thinking control, prepend `/no_think` to the prompt when thinking is off (Qwen3.5 convention)

**Frontend changes (`chat.js`, `index.html`):**

1. Add a thinking toggle button next to the send button
   - Brain icon, toggles on/off
   - Visual state: purple glow when on, dim when off
   - Default: off (for speed), user enables when they want deep reasoning
   - Persisted per-session (stored in session metadata or localStorage)

2. Thinking indicator during generation — show "Thinking..." with token count in the thinking block header as it streams

### Part D: Frontend Performance

**Token rendering optimization (`chat.js`):**

1. **Batch DOM updates** — accumulate tokens for 16ms (one animation frame), then render
   - Current: every WebSocket message triggers innerHTML parse + markdown render
   - New: buffer tokens, render on requestAnimationFrame
   - During fast streaming (>10 t/s), this prevents jank

2. **Deferred markdown** — during streaming, render as plain text with basic formatting
   - Only parse full markdown on `done` or after 500ms pause
   - Keeps the UI responsive during fast token delivery

3. **Throttle thinking block rendering** — thinking content can be hundreds of tokens
   - Render thinking block at most every 200ms during streaming
   - Final render on `thinking_end`

**Resource monitor optimization (`resources.js`):**

1. **Adaptive polling** — 2s during idle, 5s during active generation
   - Detect streaming state from app state
   - Reduces HTTP requests competing with token delivery

2. **Skip GPU poll during streaming** — GPU info is slow to query via pynvml
   - Cache GPU info for 5s during active generation instead of 1s

### Part E: UI/UX Improvements

**Chat area:**

1. **Empty state** — when no chat is selected, show a centered welcome message with:
   - App logo/name
   - "Select a chat or create one to get started"
   - Quick-start hint for keyboard shortcut

2. **Message timestamps** — subtle relative timestamps ("2m ago") on hover

3. **Copy button** — on hover over assistant messages, show a copy-to-clipboard icon

4. **Auto-title** — after first assistant response, auto-generate a chat title from the conversation (use first ~50 chars of user message, or ask the model to title it)

**Sidebar:**

1. **Keyboard shortcuts**:
   - `Ctrl+N` — new chat
   - `Ctrl+Shift+N` — new project
   - `Escape` — close modals

2. **Session search** — small search/filter input at top of session list for finding chats by title

3. **Active session persistence** — remember last active session in localStorage, restore on page load

**Model section:**

1. **TPS display** — show current generation speed in the model status area during inference
   - Updates live as tokens stream in
   - Shows "15.2 t/s" next to the green dot

2. **VRAM usage in model status** — compact display: "5.1/8.0 GB" below model name when loaded

**Visual polish:**

1. **Smooth transitions** — add transition to session list re-renders (currently jumps)
2. **Focus rings** — consistent focus-visible outlines for accessibility
3. **Loading skeletons** — pulse animation placeholders while session list loads

### Part F: Bug Fixes (discovered during review)

1. **`_kill_server_on_port` needs elevated permissions on Windows** — `psutil.net_connections()` may fail without admin. Add fallback using `netstat` parsing.

2. **Chat input stays disabled after model loads** — `setEnabled()` is only called in `_switchSession`, not when model status changes. If user creates a chat before loading a model, input stays disabled even after load.

3. **aiohttp session leak** — `_get_http_session()` in chat.py creates a module-level session that's never closed on shutdown. Add cleanup.

4. **Thinking block regex fragile** — `<thinking>` tags in stored messages could match user-typed content. Use a more specific delimiter or store thinking separately.

## Files Modified

| File | Changes |
|------|---------|
| `app/services/model_manager.py` | Command flags, `_compute_optimal_config()`, remove `--no-mmap` |
| `app/routers/chat.py` | Thinking mode control, request params |
| `static/js/chat.js` | Token batching, deferred markdown, thinking toggle, copy button, auto-title |
| `static/js/app.js` | Keyboard shortcuts, session persistence, empty state, search |
| `static/js/models.js` | TPS display, VRAM display, auto-config recommendations |
| `static/js/resources.js` | Adaptive polling |
| `static/css/style.css` | Thinking toggle, empty state, transitions, focus rings, TPS display |
| `static/index.html` | Thinking toggle button, search input, empty state markup |

## Testing Strategy

- Benchmark 9B model: measure t/s before and after flag changes
- Verify 16K and 32K context work without OOM
- Test thinking toggle on/off with Qwen3.5
- Test streaming at high t/s (4B model) to verify no jank
- Verify model load time improvement (no-mmap removal)
- Test unload/reload cycle with new flags
- Keyboard shortcut testing across browsers
