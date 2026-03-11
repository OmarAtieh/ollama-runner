# Performance Optimization & UX Improvements

**Date:** 2026-03-11
**Status:** Draft v2

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

3. **Thread priority** — add `--prio 1` (medium, not high)
   - Avoid `--prio 2` (high) as it may require elevated permissions on Windows
   - Medium priority still reduces scheduling latency
   - If flag fails, it is non-fatal — llama-server ignores unsupported priorities

4. **Thread tuning** — add `-tb` (threads-batch) = physical core count
   - Use all physical cores for prompt processing (batch)
   - Generation threads stay at physical - 1 (current behavior, correct)

5. **Single slot** — add `-np 1`
   - Single user app, no need for parallel slots
   - Reduces memory overhead and scheduling complexity

6. **Keep `--no-mmap` on Windows** — existing code is correct
   - On Windows + CUDA, mmap interferes with CUDA pinned memory transfers
   - `--mlock` solves a different problem (preventing swapout)
   - Both flags serve different purposes and should both remain

7. **Continuous batching** — explicitly enable `-cb`
   - Should be default in server mode but explicit is safer

**Flags applied incrementally during testing** — benchmark t/s and TTFT after each group:
- Group 1: KV cache quantization (biggest expected impact)
- Group 2: Batch size + thread tuning
- Group 3: Priority + slot count

### Part B: Smart Defaults & Auto-Configuration

**New method `_compute_optimal_config()` in `model_manager.py`:**

Given a model's file size, layer count, embedding dimension, and KV head count, plus current VRAM/RAM:

1. **Auto GPU layers**: Calculate max layers that fit after reserving VRAM for KV cache at target context
2. **Auto context**: If all layers fit on GPU, maximize context with remaining VRAM. Cap at model's native max.
3. **KV cache budget formula** (correct for GQA/MQA models):
   ```
   kv_bytes_per_layer = 2 * num_kv_heads * head_dim * context_length * bytes_per_element
   total_kv_bytes = num_layers * kv_bytes_per_layer

   With quantized KV cache:
     K (q8_0): bytes_per_element ≈ 1.0625
     V (q4_0): bytes_per_element ≈ 0.5625
     Average: ~0.8 bytes per element

   For Qwen3.5-9B (4096 embed, 8 KV heads, 128 head_dim, 32 layers):
     At 16K ctx: 32 * 2 * 8 * 128 * 16384 * 0.8 ≈ 860MB (f16 would be ~1.6GB)
     At 8K ctx:  ~430MB
   ```
4. **Read num_kv_heads from GGUF metadata** — add `num_kv_heads` and `num_attention_heads` to `GGUFMetadata` in `model_scanner.py`. Fall back to `num_attention_heads` if KV heads not specified (non-GQA model).
5. **Expose as "Recommended" in UI** — user can override but defaults are optimal

### Part C: Thinking Mode Control

**Mechanism: `/no_think` prompt prefix (primary, works with llama-server)**

llama-server's OpenAI-compatible API does not support `enable_thinking` — that is a vLLM/Qwen-specific extension. The only reliable mechanism with llama-server + Qwen3.5 is the `/no_think` text prefix convention.

**Backend changes (`chat.py`):**

1. Add `thinking` field to WebSocket message protocol:
   - Client sends: `{"content": "hello", "thinking": false}`
   - `chat.py` reads `data.get("thinking", False)`
   - When `thinking` is false, prepend `/no_think\n` to the user message content before sending to llama-server
   - When `thinking` is true, send as-is (Qwen3.5 defaults to thinking-on)
   - The `/no_think` prefix is NOT stored in the session DB — only the actual user message is saved

2. Thinking state is stored per-session in **localStorage** on the frontend (key: `thinking_${sessionId}`). Not stored in the DB — it's a UI preference, not conversation data.

**Frontend changes (`chat.js`, `index.html`):**

1. Thinking toggle button next to the send button:
   - Brain icon (`🧠`), toggles on/off
   - Purple glow when on, dim muted when off
   - Default: off
   - Tooltip: "Enable deep reasoning (slower)"
   - State persisted in localStorage per session

2. Send includes thinking state:
   ```js
   this.api.sendMessage(content, this._thinkingEnabled);
   ```

3. WebSocket `sendMessage` updated:
   ```js
   sendMessage(content, thinking = false) {
       this.ws.send(JSON.stringify({ content, thinking }));
   }
   ```

### Part D: Frontend Performance

**Token rendering optimization (`chat.js`):**

1. **Batch DOM updates via requestAnimationFrame**
   - Current: every WebSocket message triggers innerHTML + markdown render
   - New: buffer tokens in a string, schedule render via `requestAnimationFrame`
   - `_pendingTokens` accumulates, `_renderFrame()` flushes and renders
   - During fast streaming (>10 t/s), this batches multiple tokens per frame

2. **Deferred markdown** — during streaming, render as escaped HTML with line breaks only
   - Full markdown parse only on `done` message
   - Eliminates repeated expensive marked.parse() calls during streaming
   - Simple inline formatting (bold, code) can be applied via lightweight regex if desired

3. **Throttle thinking block rendering** — render at most every 200ms during streaming
   - Use a `_thinkingRenderTimer` to debounce
   - Final render on `thinking_end`

**Resource monitor optimization (`resources.js`):**

1. **Adaptive polling** — 2s default, 5s during active generation
   - `ResourceMonitor` exposes `setStreaming(bool)` method
   - `app.js` calls `this.resources.setStreaming(true)` when streaming starts (on `start` WS message), `false` on `done`/`error`
   - When streaming: poll every 5s instead of 2s

2. **GPU cache duration during streaming** — extend `SystemMonitor._gpu_cache` TTL from 1s to 5s during active generation (backend side, controlled by a flag on the `/api/system/resources` endpoint: `?fast=true` skips GPU re-query if cache is <5s old)

### Part E: UI/UX Improvements

**Chat area:**

1. **Empty state** — when no chat is selected, show centered:
   - App name in accent color
   - "Start a conversation" subtitle
   - Hint: "Press Ctrl+N to create a new chat"

2. **Copy button** — on hover over assistant messages, show a small copy icon in top-right corner
   - Copies the text content (not HTML) to clipboard
   - Brief "Copied!" feedback

3. **Auto-title** — use first user message, truncated to 50 chars
   - Simple and reliable, no model inference needed (avoids slot contention with `-np 1`)
   - Applied after first user message is sent
   - User can still rename via double-click

**Sidebar:**

1. **Keyboard shortcuts**:
   - `Ctrl+N` — new chat
   - `Escape` — close modals/context menus

2. **Active session persistence** — store `activeSessionId` in localStorage
   - On page load, restore last active session and load its messages
   - If session no longer exists, clear gracefully

**Model section:**

1. **TPS display** — show live generation speed in model status area
   - During streaming: "12.3 t/s" updated every 500ms based on token count / elapsed time
   - After generation: shows final TPS from stats
   - Uses the `model-status-text` element, replaces "Ready" during generation

2. **VRAM in status** — show "5.1/8.0G" below model name when loaded
   - Pulled from resource monitor data, formatted compactly

**Visual polish:**

1. **Focus rings** — add `outline: 2px solid var(--accent)` on `:focus-visible` for all interactive elements
2. **Smooth session list** — wrap session list changes in a CSS `transition: opacity 0.15s` to avoid hard jumps

### Part F: Bug Fixes

1. **Orphaned server kill fallback** — `_kill_server_on_port()` uses `psutil.net_connections()` which may require elevated permissions on Windows. Add fallback: parse `netstat -ano` output to find PID, then kill via `taskkill`.

2. **Chat input stays disabled** — `chat.setEnabled()` is only called in `_switchSession`. Add a check in `models.refreshStatus()`: if a model becomes loaded and there's an active session, call `chat.setEnabled(true)`.

3. **Thinking content stored with `<thinking>` tags** — fragile if user types `<thinking>` in chat. Fix: store reasoning in a separate `reasoning_content` field in the messages table. Add migration: `ALTER TABLE messages ADD COLUMN reasoning_content TEXT DEFAULT ''`. Update `chat.py` to save reasoning separately, update `chat.js` to read it from message history.

4. **aiohttp session cleanup** — add shutdown handler in `main.py` to close the module-level aiohttp session in `chat.py`.

## Files Modified

| File | Changes |
|------|---------|
| `app/services/model_manager.py` | Command flags, `_compute_optimal_config()`, `_kill_server_on_port` fallback |
| `app/services/model_scanner.py` | Read `num_kv_heads` from GGUF metadata |
| `app/services/session_manager.py` | Add `reasoning_content` column + migration |
| `app/services/system_monitor.py` | GPU cache TTL parameter for streaming mode |
| `app/routers/chat.py` | Thinking mode, reasoning storage, aiohttp cleanup |
| `app/main.py` | Shutdown handler for aiohttp session |
| `static/js/chat.js` | Token batching, deferred markdown, thinking toggle, copy button, auto-title, TPS display |
| `static/js/app.js` | Keyboard shortcuts, session persistence, empty state, chat enable on model load |
| `static/js/models.js` | VRAM display, auto-config recommendations |
| `static/js/api.js` | Update sendMessage signature |
| `static/js/resources.js` | Adaptive polling, setStreaming method |
| `static/css/style.css` | Thinking toggle, empty state, copy button, focus rings, TPS display |
| `static/index.html` | Thinking toggle button, empty state markup |

## Testing Strategy

**Performance benchmarking (incremental):**
1. Baseline: load 9B with current flags, measure t/s and TTFT on a standard prompt
2. Apply Group 1 (KV quant), re-benchmark
3. Apply Group 2 (batch + threads), re-benchmark
4. Apply Group 3 (priority + slots), re-benchmark
5. Test 16K and 32K context — verify no OOM, measure TTFT impact

**Functional testing:**
- Thinking toggle: verify `/no_think` suppresses reasoning, toggle on produces thinking block
- Auto-title: verify title updates after first message
- Session persistence: reload page, verify session restores
- Copy button: verify clipboard content
- Keyboard shortcuts: Ctrl+N creates chat, Escape closes modals
- Token batching: test with 4B model (fastest) to verify no visual glitches at high t/s

**Edge cases:**
- Load model → refresh page → verify state recovery
- Unload orphaned server → verify kill works without admin
- Stream with thinking on → close browser → reopen → verify stored message renders correctly
