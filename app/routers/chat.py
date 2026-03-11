"""WebSocket chat endpoint with SSE streaming from llama-server."""

import json
import logging
import time

import aiohttp
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.services.model_manager import ModelManager
from app.services.prompt_composer import PromptComposer
from app.services.session_manager import SessionManager

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Module-level aiohttp session for connection reuse
_http_session: aiohttp.ClientSession | None = None


async def _get_http_session() -> aiohttp.ClientSession:
    """Lazy-init a module-level aiohttp ClientSession."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    return _http_session


_session_manager: SessionManager | None = None


async def get_session_manager() -> SessionManager:
    """Dependency that returns the singleton SessionManager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        await _session_manager.init_db()
    return _session_manager


def get_model_manager() -> ModelManager:
    """Dependency that returns the singleton ModelManager."""
    return ModelManager.instance()


@router.websocket("/ws/chat/{session_id}")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
    mm: ModelManager = Depends(get_model_manager),
):
    """WebSocket endpoint that proxies SSE from llama-server token by token."""
    await websocket.accept()

    composer = PromptComposer()

    try:
        while True:
            # Wait for user message
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                user_content = data.get("content", "")
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_json({"type": "error", "content": "Invalid JSON"})
                continue

            if not user_content:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue

            # Check if model is loaded
            status = mm.get_status()
            if not status["loaded"]:
                await websocket.send_json({"type": "error", "content": "No model loaded"})
                continue

            server_url = status["server_url"]

            # Save user message to session
            await sm.add_message(session_id, "user", user_content)

            # Build messages array from session history + system prompt
            system_prompt = composer.compose_system_prompt()
            history = await sm.get_messages(session_id)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})

            # Send start signal
            await websocket.send_json({"type": "start"})

            # Get model config for generation params
            model_config = None
            if status.get("current_model"):
                model_config = status["current_model"]

            gen_params = {
                "messages": messages,
                "stream": True,
            }
            if model_config:
                if model_config.get("temperature") is not None:
                    gen_params["temperature"] = model_config["temperature"]
                if model_config.get("top_p") is not None:
                    gen_params["top_p"] = model_config["top_p"]
                if model_config.get("repeat_penalty") is not None:
                    gen_params["repetition_penalty"] = model_config["repeat_penalty"]

            # Stream from llama-server
            full_response = ""
            full_reasoning = ""
            token_count = 0
            t_request_start = time.perf_counter()
            t_first_token = None
            in_thinking = False

            try:
                http = await _get_http_session()
                async with http.post(
                    f"{server_url}/v1/chat/completions",
                    json=gen_params,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        await websocket.send_json(
                            {"type": "error", "content": f"Server error: {resp.status} {error_text}"}
                        )
                        continue

                    # Read SSE stream line by line
                    async for line_bytes in resp.content:
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue

                        payload = line[6:]  # strip "data: "

                        if payload == "[DONE]":
                            break

                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})

                        # Handle reasoning/thinking content (e.g. Qwen3.5, DeepSeek-R1)
                        reasoning = delta.get("reasoning_content")
                        if reasoning:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            token_count += 1
                            full_reasoning += reasoning

                            # Show thinking in UI with a distinct marker
                            if not in_thinking:
                                in_thinking = True
                                await websocket.send_json({"type": "thinking_start"})
                            await websocket.send_json({"type": "thinking", "content": reasoning})
                            continue

                        content = delta.get("content")
                        if content is None:
                            continue

                        # Transition from thinking to content
                        if in_thinking:
                            in_thinking = False
                            await websocket.send_json({"type": "thinking_end"})

                        # Track timing
                        if t_first_token is None:
                            t_first_token = time.perf_counter()

                        token_count += 1
                        full_response += content

                        # Send token immediately - zero buffering
                        await websocket.send_json({"type": "token", "content": content})

            except aiohttp.ClientError as e:
                await websocket.send_json(
                    {"type": "error", "content": f"Connection error: {e}"}
                )
                continue

            # Close any open thinking block
            if in_thinking:
                await websocket.send_json({"type": "thinking_end"})

            # Calculate stats
            t_end = time.perf_counter()

            ttft_ms = 0.0
            if t_first_token is not None:
                ttft_ms = (t_first_token - t_request_start) * 1000

            tps = 0.0
            if t_first_token is not None and token_count > 1:
                generation_time = t_end - t_first_token
                if generation_time > 0:
                    tps = token_count / generation_time

            stats = {
                "token_count": token_count,
                "tokens_per_second": round(tps, 1),
                "time_to_first_token_ms": round(ttft_ms, 1),
            }

            # Build saved content: include reasoning if present
            saved_content = full_response
            if full_reasoning:
                saved_content = f"<thinking>\n{full_reasoning}\n</thinking>\n\n{full_response}"

            # Send done message
            await websocket.send_json({
                "type": "done",
                "content": full_response,
                "reasoning": full_reasoning if full_reasoning else None,
                "stats": stats,
            })

            # Save assistant message with stats (include reasoning in stored content)
            await sm.add_message(
                session_id,
                "assistant",
                saved_content,
                token_count=token_count,
                tokens_per_second=round(tps, 1),
                time_to_first_token_ms=round(ttft_ms, 1),
            )

    except WebSocketDisconnect:
        log.info("WebSocket disconnected for session %s", session_id)
    except Exception as e:
        log.error("WebSocket error for session %s: %s", session_id, e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass
