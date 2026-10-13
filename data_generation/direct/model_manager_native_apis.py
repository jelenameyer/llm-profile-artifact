#!/usr/bin/env python3
"""Direct generation path (proprietary models) — entry point.

Runs proprietary LLMs on the instrument items directly (no-context only) across
reasoning conditions (no-reasoning / low-reasoning), via each provider's native
API. All generation parameters are defined once (see EXPERIMENT CONFIGURATION
below) and applied identically wherever the API permits; every parameter actually
sent is recorded in the output.

Providers (model-id prefix → API):
  anthropic/  → Anthropic Messages API
  openai/     → OpenAI Chat Completions API
  google/     → Google Generative Language API
  qwen/       → Alibaba DashScope (OpenAI-compatible endpoint)
  x-ai/       → xAI API (OpenAI-compatible endpoint)
  vllm/       → self-hosted vLLM server (OpenAI-compatible; --vllm-base-url)

Reads the task definitions in <task-dir>/ (+ their tasks/jsonl_data/*.jsonl items).
Writes <outputs-dir>/<safe-model>__<condition>_<task>_results.csv and a run log,
under outputs_api/.

Usage: python model_manager_native_apis.py --models-file api_models.txt

The large open-weight models were served on a self-hosted vLLM instance, one at a
time, so each was run separately with its own --vllm-base-url and --vllm-key-env:

    python model_manager_native_apis.py \
        --models vllm/<org>/<model> \
        --vllm-base-url https://<host>/v1 --vllm-key-env <ENV_VAR_NAME> \
        --task-filter barratt,care,dast,dfd,dm,dospert,gabs,ipipneo,lot,mpl,pri,soep,sssv \
        --timeout 600 --outputs-dir outputs_api_<model>

--task-filter is required for those runs: discover_tasks() globs every *.py in
tasks/, which otherwise pulls in the *_flipped.py variants that the proprietary
panel never ran.
"""

import abc
import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, parse, request

import pandas as pd
import base_functions


# ── Logging ────────────────────────────────────────────────────────────────

os.makedirs("outputs_api", exist_ok=True)
# The PID is part of the name because the large open-weight models are served on
# separate endpoints and are therefore collected in PARALLEL, one process per
# model. The timestamp alone has second resolution, so processes launched
# together would share one log file, append to it concurrently and tear each
# other's lines -- and this file is the verbatim request/response record that
# gets archived, not a convenience.
log_file = os.path.join(
    "outputs_api",
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log",
)
# The file handler records at DEBUG so that the request payload and the full
# response body of every call are kept verbatim; the console stays at INFO so
# the run remains readable. Without this the payload logging in _http_post is
# emitted at DEBUG and then discarded by an INFO-level root logger.
_file_handler = logging.FileHandler(log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[_file_handler, _stream_handler],
)


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION — all tunable parameters are defined here
# ══════════════════════════════════════════════════════════════════════════

# Reasoning conditions: (label_suffix, reasoning_effort_value)
# None = omit reasoning parameter entirely
REASONING_CONDITIONS: List[Tuple[str, Optional[str]]] = [
    ("nr", None),      # no reasoning
    ("r",  "low"),    # reasoning  (disabled for the non-ipipneo task batch)
]

# Per-model overrides: only the listed conditions are run for matching model substrings.
# Key is matched as a substring of the full model name.
MODEL_CONDITION_OVERRIDES: Dict[str, List[str]] = {
    "grok-4.20-0309-reasoning": ["r"],   # reasoning variant → reasoning condition only
    "grok-4.20-0309-non-reasoning":           ["nr"],  # non-reasoning variant → nr condition only
}

# Shared generation parameters — identical across all providers where supported.
# Exceptions that cannot be reconciled:
#   - Anthropic + Qwen force temperature=1.0 when thinking is enabled
#   - Anthropic omits top_p when thinking is enabled (API rejects it)
MAX_OUTPUT_TOKENS    = 16
MAX_REASONING_TOKENS = 1_024
MAX_TOKENS_REASONING = MAX_REASONING_TOKENS + MAX_OUTPUT_TOKENS  # 1040 for OpenAI, XAI we need to give total number reasoning+output

GEN_PARAMS = dict(
    max_tokens     = MAX_OUTPUT_TOKENS,
    temperature    = 0.0,
    top_p          = 1.0,
    hide_reasoning = False,
)

GEN_PARAMS_REASONING = dict(
    max_tokens     = MAX_OUTPUT_TOKENS,
    temperature    = 1.0,       # required by Anthropic; recommended by Qwen; omitted for OpenAI (see client)
    top_p          = 1.0,
    hide_reasoning = False,
)

# Thinking budget in tokens for each effort level.
_THINKING_BUDGET_TOKENS = MAX_REASONING_TOKENS

THINKING_BUDGET: Dict[str, int] = {
    "none": 0,
    "low": _THINKING_BUDGET_TOKENS,
}

# Fallback label used in error records when reasoning effort is not in REASONING_CONDITIONS.
UNKNOWN_CONDITION_LABEL = "unknown"

# Per-model minimum delay in seconds between requests (for haiku otherwise too many requests in too short time, error).
# Keyed by substring match against full model name.
# 60s / 50rpm = 1.2s minimum; use 1.5s for safety margin.
MODEL_REQUEST_DELAY_S: Dict[str, float] = {
    "haiku": 1.5,
}

# Per-provider replacements for GEN_PARAMS_REASONING. The default sets
# temperature=1.0 only because Anthropic and Qwen force it under extended
# thinking; a self-hosted vLLM server has no such constraint, so the reasoning
# condition is run greedily there.
#
# NOTE: greedy does NOT mean reproducible. Identical requests at temperature 0
# returned different chains of thought, and sometimes different answers, on
# several of these models -- continuous batching changes floating-point reduction
# order, and the difference compounds over hundreds of tokens. So the
# single-observation caveat in SI A.10 applies to the reasoning condition too.
GEN_PARAMS_REASONING_OVERRIDES: Dict[str, dict] = {
    "vllm": dict(GEN_PARAMS_REASONING, temperature=0.0),
}

# Serving details no API reports, recorded by hand. Keys are matched as a
# SUBSTRING of the model name, in insertion order, and the last match wins -- so a
# more specific repo id must come after a less specific one. Values are merged into
# the per-call metadata and become columns in the output CSV.
# --served-quantization overrides these at run time.
#
# Read the dtype off the checkpoint, not off vLLM's default: vLLM honours a
# checkpoint's own quantization_config, so "no --quantization flag" does not mean
# bf16. Each model below is recorded at the precision it was released in.
MODEL_SERVING_NOTES: Dict[str, Dict[str, str]] = {
    "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {"served_quantization": "bf16"},
    "MiniMax-M3": {"served_quantization": "bf16"},
    # FP8 checkpoint, despite the unqualified repo name; a separate -BF16 repo
    # exists and was not the one served.
    "GLM-5.3-Flash": {"served_quantization": "fp8-e4m3"},
    "GLM-5.3-Flash-BF16": {"served_quantization": "bf16"},
    # Here the unqualified repo IS bf16, but official -FP8, -NVFP4 and -GPTQ-Int4
    # variants exist; add the longer id after this entry if one is ever served.
    "Qwen3.5-397B-A17B": {"served_quantization": "bf16"},
    # Native INT4 (compressed-tensors, 4-bit group-32 on the routed experts). No
    # higher-precision sibling is published.
    "Kimi-K2.6": {"served_quantization": "int4-native"},
}

# Finish reasons meaning the model was still generating when the cap was hit.
TRUNCATED_FINISH_REASONS = {"length", "MAX_TOKENS", "max_tokens"}

# Bump whenever the way a request is BUILT changes (generation parameters, the
# thinking on/off mechanism, the prompt construction). It is part of the response
# cache key, so bumping it invalidates cached answers that were collected under
# the old protocol instead of silently replaying them.
CACHE_PROTOCOL_VERSION = 5

# Models whose chat template hardcodes an OPEN <think> into the generation prompt,
# with no branch that omits it, so no request parameter can suppress reasoning.
# For these the non-reasoning condition is built by prefilling the assistant turn
# with an already-CLOSED thinking block and continuing from it.
#
# Add a model here ONLY if the assistant-prefill request is the one that returns
# zero characters of thinking. Every other template in this panel closes the block
# itself, and prefilling those as well would emit the closing tag twice.
VLLM_THINK_PREFILL_MODELS: Tuple[str, ...] = (
    "GLM-5.3-Flash",
)
THINK_PREFILL_CONTENT = "<think></think>"

# Effort sent in the NON-reasoning condition for the prefill models above. Their
# reasoning_effort has no "off" level and resolves to max when omitted, which would
# stamp "Reasoning Effort: Max" into the prompt of the condition where reasoning is
# being suppressed. The lowest available level is the least wrong option.
VLLM_NONREASONING_EFFORT = "low"

# Effort used in the REASONING condition where the panel-wide "low" does not mean
# "a smaller budget". On GLM-5.3-Flash "low" suppresses reasoning outright, so it
# cannot serve as the reasoning condition; "max" is that model's own default.
VLLM_REASONING_EFFORT: Dict[str, str] = {
    "GLM-5.3-Flash": "max",
}

# Which chat-template kwarg controls thinking, per model, as
# (kwarg_name, value_when_reasoning, value_when_not). enable_thinking is only a
# convention, not a standard: a template that does not recognise the kwarg IGNORES
# it rather than erroring, and the failure is silent and lands in the non-reasoning
# condition -- the template falls through to its own default, the model reasons
# anyway, and under the 16-token cap the result is a truncated thought with no
# answer. So read each model's chat_template.jinja before collecting it.
#
# None of the models below belongs in VLLM_THINK_PREFILL_MODELS: their templates
# emit the closed thinking block themselves on the "off" value.
VLLM_THINKING_KWARG: Dict[str, Tuple[str, object, object]] = {
    # thinking_mode: "enabled" / "disabled" / "adaptive"; an undefined value falls
    # through to adaptive, i.e. no off switch.
    "MiniMax-M3": ("thinking_mode", "enabled", "disabled"),
    # A third spelling: plain boolean `thinking`.
    "Kimi-K2.6": ("thinking", True, False),
    # Qwen3.5-397B-A17B has NO entry on purpose: it reads the default below.
}
VLLM_THINKING_KWARG_DEFAULT: Tuple[str, object, object] = ("enable_thinking", True, False)


def _thinking_kwarg(model_name: str) -> Tuple[str, object, object]:
    """The (kwarg, on_value, off_value) triple that controls thinking."""
    for pattern, spec in VLLM_THINKING_KWARG.items():
        if pattern in model_name:
            return spec
    return VLLM_THINKING_KWARG_DEFAULT


def _template_forces_thinking(model_name: str) -> bool:
    """True for models whose chat template cannot be talked out of reasoning."""
    return any(p in model_name for p in VLLM_THINK_PREFILL_MODELS)

# Open-weight reasoning models return the chain of thought INLINE in the message
# content, closed by a think tag, unless the server runs a --reasoning-parser. The
# answer is whatever follows the FINAL closing tag; matching against the whole
# string instead picks up digits out of the reasoning and yields a valid-looking
# but wrong answer for every item.
#
# </mm:think> is MiniMax's tag and is NOT a substring match for </think>, so it has
# to be listed separately.
_THINK_CLOSE_RE = re.compile(
    r"</think>|</thinking>|</mm:think>|<\|end_of_thought\|>", re.IGNORECASE
)


def _split_inline_reasoning(text: str) -> Tuple[str, Optional[str]]:
    """Split 'reasoning</think>answer' into (answer_text, reasoning_text).

    Returns (text, None) when no closing tag is present, i.e. for providers that
    return reasoning separately or for models that do not reason at all.
    """
    if not text:
        return text, None
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if not matches:
        return text, None
    last = matches[-1]
    return text[last.end():].strip(), text[:last.start()].strip()


# ── Utilities ──────────────────────────────────────────────────────────────

def _safe_model_key(model_name: str) -> str:
    key = model_name.strip().replace("/", "__").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def _http_get(url: str, headers: dict, timeout_s: int) -> dict:
    req = request.Request(url, headers=headers, method="GET")
    with request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def preflight_vllm(base_url: str, api_key: Optional[str], timeout_s: int,
                   requested_models: List[str]) -> None:
    """Check the server is reachable and serving the models we are about to run.

    Without this a malformed URL or an unloaded model costs a full retry cascade
    per item before the run aborts, and the underlying error is buried in the
    retry warnings.
    """
    if base_url.count("://") > 1:
        raise RuntimeError(
            f"--vllm-base-url looks doubled: {base_url!r}. "
            "Expected a single address such as https://host/path/v1"
        )
    parsed_url = parse.urlparse(base_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        raise RuntimeError(
            f"--vllm-base-url is not a valid http(s) URL: {base_url!r}"
        )

    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key or 'EMPTY'}"}
    try:
        parsed = _http_get(url, headers, min(timeout_s, 30))
    except error.HTTPError as exc:
        # The server answered, so the address is right; the request was refused.
        body = ""
        try:
            body = exc.read().decode("utf-8")[:300]
        except Exception:
            pass
        hint = {
            401: "The server rejected the token. Check the key is exported in this "
                 "shell (echo ${VLLM_API_KEY:+set}) and matches the token you were "
                 "given; use --vllm-key-env if it lives under another variable name.",
            403: "Authenticated but not permitted to use this endpoint.",
            404: "No /models endpoint at this path. Check whether the base URL "
                 "should end in /v1.",
        }.get(exc.code, "")
        raise RuntimeError(
            f"vLLM server at {url} returned HTTP {exc.code} ({exc.reason}). "
            f"{hint}\nResponse body: {body}"
        ) from exc
    except json.JSONDecodeError as exc:
        # Self-hosted deployments often sit behind a web front-end. When a
        # deployment expires, its path stops routing to vLLM and falls through to
        # that app's catch-all route, which answers HTTP 200 with an HTML shell --
        # so a status-code-only health check reports success on a dead endpoint.
        # Hence this check parses the body. Observed 30 Sep 2026.
        raise RuntimeError(
            f"{url} answered with HTML, not JSON, so this path is not a live vLLM "
            "server. The usual cause is an expired deployment: the serving session "
            "has ended and the URL now resolves to the portal's web app, which "
            "returns HTTP 200 for any path. Ask for the current deployment URL and "
            "token. (A malformed base URL produces the same symptom.)"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach the vLLM server at {url}: {exc}\n"
            "Check that the model has finished loading and that --vllm-base-url "
            "is the server address including the /v1 suffix."
        ) from exc

    served = [entry.get("id") for entry in (parsed.get("data") or [])]
    logging.info("vLLM server reachable; serving: %s", served)

    for full_name in requested_models:
        native_id = full_name.split("/", 1)[1]
        if served and native_id not in served:
            raise RuntimeError(
                f"Model {native_id!r} is not served at this address. "
                f"Available: {served}"
            )


def _http_post(url: str, payload: dict, headers: dict, timeout_s: int) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    # Sanitize sensitive fields before logging
    safe_url = re.sub(r"key=[^&]+", "key=***", url)
    safe_headers = {
        k: "***" if k.lower() in ("x-api-key", "authorization") else v
        for k, v in headers.items()
    }
    logging.debug(
        "API REQUEST →\nURL: %s\nHeaders: %s\nPayload: %s",
        safe_url,
        json.dumps(safe_headers, indent=2),
        json.dumps(payload, indent=2),
    )

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    logging.debug("API RESPONSE ←\nURL: %s\nBody: %s", safe_url, body)
    return json.loads(body), ts, safe_url


# ── Base client ────────────────────────────────────────────────────────────

class BaseAPIClient(abc.ABC):
    @abc.abstractmethod
    def chat_completion(
        self,
        model_name: str,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: Optional[str],
        hide_reasoning: bool,
    ) -> Tuple[str, dict]:
        """Returns (text_content, meta_dict)."""
        ...


# ── Anthropic ──────────────────────────────────────────────────────────────

class AnthropicClient(BaseAPIClient):
    """
    Anthropic Messages API.
    https://docs.anthropic.com/en/api/messages

    Constraints that cannot be worked around:
      - temperature MUST be 1.0 when extended thinking is enabled.
      - top_p is mutually exclusive with temperature (Anthropic rejects both).
        We send top_p only when thinking is disabled.
      - top_k: supported but omitted here for cross-provider comparability.
    """
    BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, api_key: str, timeout_s: int = 120):
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_completion(
        self,
        model_name: str,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: Optional[str],
        hide_reasoning: bool,
    ) -> Tuple[str, dict]:

        system_text: Optional[str] = None
        filtered = []
        for m in messages:
            # Claude does not accept a system role inside the messages array, so separate it out manually
            if m["role"] == "system":
                system_text = m["content"]
            else:
                filtered.append({"role": m["role"], "content": m["content"]})

        budget = THINKING_BUDGET.get(reasoning_effort or "none", 0)
        # define boolean variable, if thinking allowed or not, derived from whether token budget is bigger than 0
        thinking_enabled = budget > 0

        # max_tokens must exceed budget_tokens
        actual_max_tokens = (MAX_REASONING_TOKENS + MAX_OUTPUT_TOKENS) if thinking_enabled else max_tokens

        payload: dict = {
            "model": model_name,
            "messages": filtered,
            "max_tokens": actual_max_tokens,
        }
        if system_text:
            payload["system"] = system_text

        if thinking_enabled:
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["temperature"] = 1.0          # required; cannot match other providers
            # top_p must NOT be sent alongside temperature=1 + thinking
        else:
            payload["temperature"] = temperature

        # Record every parameter actually sent
        meta: dict = {
            "provider": "anthropic",
            "requested_model": model_name,
            "param_temperature": payload.get("temperature"),
            "param_top_p": None,
            "param_max_tokens": actual_max_tokens,
            "param_thinking_enabled": thinking_enabled,
            "param_thinking_budget": budget if thinking_enabled else None,
            "param_reasoning_effort_input": reasoning_effort,
            "temperature_forced_by_thinking": thinking_enabled,
        }
        # HTTP request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        parsed, ts, req_url = _http_post(self.BASE_URL, payload, headers, self.timeout_s)

        usage = parsed.get("usage") or {}
        meta["response_model"] = parsed.get("model")
        meta["usage_input_tokens"] = usage.get("input_tokens")
        meta["usage_output_tokens"] = usage.get("output_tokens")
        meta["usage_cache_read_tokens"] = usage.get("cache_read_input_tokens")
        meta["usage_cost"] = None  # not provided by Anthropic API
        meta["timestamp_utc"] = ts
        meta["request_url"] = req_url

        text_parts = []
        for block in parsed.get("content", []):
            t = block.get("type")
            if t == "text":
                text_parts.append(block.get("text", ""))
            elif t == "thinking" and not hide_reasoning:
                meta["thinking_content"] = block.get("thinking", "")
        meta["finish_reason"] = parsed.get("stop_reason") 

        return "".join(text_parts), meta


# ── OpenAI-compatible base ─────────────────────────────────────────────────

class OpenAICompatibleClient(BaseAPIClient):
    def __init__(self, api_key: str, base_url: str, timeout_s: int = 120):
        self.api_key   = api_key
        self.base_url  = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _build_reasoning_params(
        self, model_name: str, reasoning_effort: Optional[str]
    ) -> dict:
        if not reasoning_effort or reasoning_effort == "none":
            return {}
        return {"reasoning_effort": reasoning_effort}

    def chat_completion(
        self,
        model_name: str,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: Optional[str],
        hide_reasoning: bool,
    ) -> Tuple[str, dict]:

        use_reasoning    = bool(reasoning_effort and reasoning_effort != "none")
        effective_max_tokens = MAX_TOKENS_REASONING if use_reasoning else max_tokens

        # Compute once — used in both payload and meta.
        reasoning_params = self._build_reasoning_params(model_name, reasoning_effort)

        payload: dict = {
            "model":       model_name,
            "messages":    messages,
            "max_tokens":  effective_max_tokens,
            "n":           1,
            "stream":      False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if use_reasoning:
            payload.update(reasoning_params)

        meta: dict = {
            "provider":               self.__class__.__name__,
            "requested_model":        model_name,
            "param_temperature":      temperature,
            "param_top_p":            top_p,
            "param_max_tokens":       effective_max_tokens,
            **{f"param_{k}": v for k, v in reasoning_params.items()},
        }

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url    = f"{self.base_url}/chat/completions"
        parsed, ts, req_url = _http_post(url, payload, headers, self.timeout_s)

        usage = parsed.get("usage") or {}
        meta["response_model"]          = parsed.get("model")
        meta["response_created_unix"]   = parsed.get("created")
        meta["usage_prompt_tokens"]     = usage.get("prompt_tokens")
        meta["usage_completion_tokens"] = usage.get("completion_tokens")
        # Reasoning tokens are nested under completion_tokens_details for GPT-5.4.
        meta["usage_reasoning_tokens"]  = (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        )
        meta["usage_cost"] = None

        choice  = (parsed.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        meta["timestamp_utc"] = ts
        meta["request_url"] = req_url
        meta["finish_reason"] = choice.get("finish_reason")   # "stop", "length", "content_filter"
        reasoning_content = (choice.get("message") or {}).get("reasoning_content") or ""  
        if reasoning_content and not hide_reasoning:                                         
            meta["thinking_content"] = reasoning_content 
        return content, meta


class OpenAIClient(OpenAICompatibleClient):
    def __init__(self, api_key: str, timeout_s: int = 120):
        super().__init__(api_key, "https://api.openai.com/v1", timeout_s)

    def chat_completion(self, model_name, messages, *, max_tokens, temperature,
                        top_p, reasoning_effort, hide_reasoning):
        use_reasoning = bool(reasoning_effort and reasoning_effort != "none")
        effective_max_tokens = MAX_TOKENS_REASONING if use_reasoning else max_tokens

        if use_reasoning:
            temperature = None
            top_p = None

        reasoning_params = self._build_reasoning_params(model_name, reasoning_effort)

        payload: dict = {
            "model":                 model_name,
            "messages":              messages,
            "max_completion_tokens": effective_max_tokens,  # replaces max_tokens, since variable named differently here
            "n":                     1,
            "stream":                False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if use_reasoning:
            payload.update(reasoning_params)

        meta: dict = {
            "provider":              "openai",
            "requested_model":       model_name,
            "param_temperature":     temperature,
            "param_top_p":           top_p,
            "param_max_tokens":      effective_max_tokens,
            **{f"param_{k}": v for k, v in reasoning_params.items()},
        }

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        parsed, ts, req_url = _http_post(url, payload, headers, self.timeout_s)

        usage = parsed.get("usage") or {}
        meta["response_model"]          = parsed.get("model")
        meta["response_created_unix"]   = parsed.get("created")
        meta["usage_prompt_tokens"]     = usage.get("prompt_tokens")
        meta["usage_completion_tokens"] = usage.get("completion_tokens")
        meta["usage_reasoning_tokens"]  = (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        )
        meta["usage_cost"]    = None
        meta["timestamp_utc"] = ts
        meta["request_url"]   = req_url

        choice  = (parsed.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        meta["finish_reason"] = choice.get("finish_reason")
        reasoning_content = (choice.get("message") or {}).get("reasoning_content") or ""
        if reasoning_content and not hide_reasoning:
            meta["thinking_content"] = reasoning_content

        return content, meta


class XAIClient(OpenAICompatibleClient):
    """xAI/Grok — OpenAI-compatible wire format, different base URL.
    Reasoning is encoded in the model ID (grok-4.20 vs grok-4.20-reasoning).
    reasoning_effort parameter is unsupported and causes HTTP 400.
    """
    def __init__(self, api_key: str, timeout_s: int = 120):
        super().__init__(api_key, "https://api.x.ai/v1", timeout_s)

    def _build_reasoning_params(self, model_name: str, reasoning_effort: Optional[str]) -> dict:
        return {}

# ── Qwen (DashScope OpenAI-compatible) ─────────────────────────────────────

class QwenClient(OpenAICompatibleClient):
    """
    Alibaba DashScope — OpenAI-compatible endpoint (international).
    https://www.alibabacloud.com/help/en/model-studio/

    Reasoning is controlled via enable_thinking + thinking_budget,
    NOT via reasoning_effort. These are injected as extra top-level keys.
    temperature must be in [0, 1] when enable_thinking=False;
    set to 1.0 (fixed) when enable_thinking=True — same constraint as Anthropic.
    """
    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str, timeout_s: int = 120):
        super().__init__(api_key, self.BASE_URL, timeout_s)
        self._reasoning_model_prefixes = ()  # Qwen3 uses enable_thinking, not model prefix

    def _build_reasoning_params(self, model_name, reasoning_effort):
        if not reasoning_effort or reasoning_effort == "none":
            return {"enable_thinking": False}
        budget = THINKING_BUDGET.get(reasoning_effort, THINKING_BUDGET["low"])
        return {"enable_thinking": True, "thinking_budget": budget}

    def chat_completion(self, model_name, messages, *, max_tokens, temperature,
                        top_p, reasoning_effort, hide_reasoning):
        # Force temperature=1.0 when thinking enabled (same Anthropic constraint)
        use_reasoning = bool(reasoning_effort and reasoning_effort != "none")
        effective_temp = 1.0 if use_reasoning else temperature

        # Qwen extra params go in top-level payload, not nested
        # Delegate to parent but patch temperature after
        original_temp = temperature
        # We override by calling parent with adjusted temperature
        # (parent reads temperature from argument, so pass effective_temp)

        effective_max_tokens = MAX_TOKENS_REASONING if use_reasoning else max_tokens

        payload: dict = {
            "model": model_name,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": effective_temp,
            "top_p": top_p,
            "n": 1,
            "stream": False,
        }
        payload.update(self._build_reasoning_params(model_name, reasoning_effort))

        meta: dict = {
            "provider": "qwen",
            "requested_model": model_name,
            "param_temperature": effective_temp,
            "param_temperature_requested": original_temp,
            "param_top_p": top_p,
            "param_max_tokens": effective_max_tokens,
            "param_reasoning_effort_input": reasoning_effort,
            "param_enable_thinking": use_reasoning,
            "param_thinking_budget": THINKING_BUDGET.get(reasoning_effort, 0) if use_reasoning else None,
            "temperature_forced_by_thinking": use_reasoning,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        parsed, ts, req_url = _http_post(url, payload, headers, self.timeout_s)

        usage = parsed.get("usage") or {}
        meta["response_model"] = parsed.get("model")
        meta["usage_prompt_tokens"] = usage.get("prompt_tokens")
        meta["usage_completion_tokens"] = usage.get("completion_tokens")
        meta["usage_thinking_tokens"] = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        meta["usage_cost"] = None

        choice = (parsed.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        meta["timestamp_utc"] = ts
        meta["request_url"] = req_url       
        meta["finish_reason"] = choice.get("finish_reason")
        reasoning_content = (choice.get("message") or {}).get("reasoning_content") or ""  
        if reasoning_content and not hide_reasoning:                                         
            meta["thinking_content"] = reasoning_content
        return content, meta


# ── Self-hosted vLLM ───────────────────────────────────────────────────────

class VLLMClient(OpenAICompatibleClient):
    """
    Self-hosted vLLM OpenAI-compatible server (large open-weight models).

    Two things differ from the commercial OpenAI-compatible providers:

      - The base URL is supplied at run time (--vllm-base-url / VLLM_BASE_URL)
        rather than fixed as a class constant, because each model is loaded
        into its own short-lived serving session with its own address.

      - How thinking is controlled varies by model family, so both levers are
        used. chat_template_kwargs.enable_thinking is sent in BOTH conditions on
        purpose (the inherited implementation applies reasoning params only when
        reasoning is on, and omitting the flag leaves the template default in
        force — thinking ON for several of these models, which under a 16-token
        cap returns an empty content string). reasoning_effort is sent per
        VLLM_REASONING_EFFORT / VLLM_NONREASONING_EFFORT, because on some models
        it is the only lever that works: GLM-5.3 ignores enable_thinking
        entirely and treats reasoning_effort=low as off rather than as a
        reduced budget.

    vLLM has no thinking-budget parameter, so reasoning length is bounded by
    max_tokens (MAX_TOKENS_REASONING) rather than by a separate budget as on
    Anthropic/Google/Qwen.
    """

    def __init__(self, api_key: Optional[str], base_url: str, timeout_s: int = 120):
        # vLLM requires a bearer token to be present but ignores its value
        # unless the server was started with --api-key.
        super().__init__(api_key or "EMPTY", base_url, timeout_s)

    def _build_reasoning_params(self, model_name: str, reasoning_effort: Optional[str]) -> dict:
        enabled = bool(reasoning_effort and reasoning_effort != "none")
        kwarg, on_value, off_value = _thinking_kwarg(model_name)
        params: dict = {"chat_template_kwargs": {kwarg: on_value if enabled else off_value}}

        # Reasoning condition: the effort level the proprietary panel used (SI
        # Table S4). Non-reasoning: normally nothing, but where the effort
        # parameter has no "off" level we send the lowest rather than let it
        # default to max (VLLM_NONREASONING_EFFORT). GLM-5.3 accepts only
        # low/high/max and treats anything else as max, so REASONING_CONDITIONS
        # must keep using one of those labels.
        effort = reasoning_effort
        if enabled:
            for pattern, level in VLLM_REASONING_EFFORT.items():
                if pattern in model_name:
                    effort = level
                    break
        elif _template_forces_thinking(model_name):
            effort = VLLM_NONREASONING_EFFORT
        if effort:
            params["reasoning_effort"] = effort
        return params

    def chat_completion(self, model_name, messages, *, max_tokens, temperature,
                        top_p, reasoning_effort, hide_reasoning):
        use_reasoning = bool(reasoning_effort and reasoning_effort != "none")
        effective_max_tokens = MAX_TOKENS_REASONING if use_reasoning else max_tokens
        reasoning_params = self._build_reasoning_params(model_name, reasoning_effort)

        # Templates that force an open <think> need the non-reasoning condition
        # built by prefilling a closed thinking block (see VLLM_THINK_PREFILL_MODELS).
        think_prefill = not use_reasoning and _template_forces_thinking(model_name)
        if think_prefill:
            messages = list(messages) + [
                {"role": "assistant", "content": THINK_PREFILL_CONTENT}
            ]

        payload: dict = {
            "model":       model_name,
            "messages":    messages,
            "max_tokens":  effective_max_tokens,
            "temperature": temperature,
            "top_p":       top_p,
            "n":           1,
            "stream":      False,
        }
        payload.update(reasoning_params)   # unconditional; see class docstring

        if think_prefill:
            # Continue the prefilled assistant turn instead of opening a new one.
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False

        meta: dict = {
            "provider":                       "vllm",
            "requested_model":                model_name,
            "param_temperature":              temperature,
            "param_top_p":                    top_p,
            "param_max_tokens":               effective_max_tokens,
            "param_enable_thinking":          use_reasoning,
            # The template kwarg actually sent, which is not enable_thinking on
            # every model (see VLLM_THINKING_KWARG). Recorded per row because it
            # is the only evidence in the CSV that the condition was built the
            # way the notes claim.
            "param_thinking_kwarg":           json.dumps(
                reasoning_params.get("chat_template_kwargs"), sort_keys=True
            ),
            "param_reasoning_effort_input":   reasoning_effort,
            # What was actually transmitted, which differs from the condition's
            # input label wherever a per-model override applies.
            "param_reasoning_effort_sent":    reasoning_params.get("reasoning_effort"),
            "param_thinking_budget":          None,   # not a vLLM parameter
            "param_think_prefill":            think_prefill,
            "temperature_forced_by_thinking": False,
        }

        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        parsed, ts, req_url = _http_post(url, payload, headers, self.timeout_s)

        usage = parsed.get("usage") or {}
        meta["response_model"]          = parsed.get("model")
        meta["response_created_unix"]   = parsed.get("created")
        meta["usage_prompt_tokens"]     = usage.get("prompt_tokens")
        meta["usage_completion_tokens"] = usage.get("completion_tokens")
        meta["usage_reasoning_tokens"]  = (
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        )
        meta["usage_cost"]    = None   # self-hosted
        meta["timestamp_utc"] = ts
        meta["request_url"]   = req_url

        choice  = (parsed.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        meta["finish_reason"] = choice.get("finish_reason")
        # Present only when the server runs with a --reasoning-parser; otherwise
        # the thinking text stays inline in content.
        reasoning_content = (choice.get("message") or {}).get("reasoning_content") or ""
        if reasoning_content and not hide_reasoning:
            meta["thinking_content"] = reasoning_content

        return content, meta


# ── Google Generative Language ─────────────────────────────────────────────

class GoogleClient(BaseAPIClient):
    """
    Google Generative Language API (Gemini).
    https://ai.google.dev/api/generate-content

    Notes:
      - No seed parameter available.
      - top_k supported but omitted for cross-provider comparability.
      - thinkingBudget=0 explicitly disables thinking.
      - temperature range is [0.0, 2.0] for Gemini; we stay in [0, 1].
    """
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, timeout_s: int = 120):
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_completion(
        self,
        model_name: str,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: Optional[str],
        hide_reasoning: bool,
    ) -> Tuple[str, dict]:

        system_parts, contents = [], []
        for m in messages:
            if m["role"] == "system":
                system_parts.append({"text": m["content"]})
            else:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        budget = THINKING_BUDGET.get(reasoning_effort or "none", 0)

        gen_cfg: dict = {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "candidateCount": 1,
            "thinkingConfig": {"thinkingBudget": budget},
        }

        payload: dict = {"contents": contents, "generationConfig": gen_cfg}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        meta: dict = {
            "provider": "google",
            "requested_model": model_name,
            "param_temperature": temperature, 
            "param_top_p": top_p,
            "param_max_tokens": max_tokens,
            "param_thinking_budget": budget,
            "param_reasoning_effort_input": reasoning_effort,
        }

        url = f"{self.BASE_URL}/models/{model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        parsed, ts, req_url = _http_post(url, payload, headers, self.timeout_s)

        usage = parsed.get("usageMetadata") or {}
        meta["response_model"] = model_name  # Google doesn't echo model in response
        meta["usage_prompt_tokens"] = usage.get("promptTokenCount")
        meta["usage_completion_tokens"] = usage.get("candidatesTokenCount")
        meta["usage_thinking_tokens"] = usage.get("thoughtsTokenCount")
        meta["usage_cost"] = None
        meta["timestamp_utc"] = ts
        meta["request_url"] = req_url
        candidates = parsed.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates. Response: {json.dumps(parsed)[:500]}")

        candidate = candidates[0]
        meta["finish_reason"] = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", [])

        text_parts, thinking_parts = [], []
        for p in parts:
            if "text" not in p:
                continue
            if p.get("thought", False):
                thinking_parts.append(p.get("text", ""))
            else:
                text_parts.append(p.get("text", ""))

        if not hide_reasoning and thinking_parts:
            meta["thinking_content"] = "".join(thinking_parts)

        text = "".join(text_parts)
        return text, meta


# ── Routing ────────────────────────────────────────────────────────────────

class RoutingClient:
    """
    Maps 'provider/model-name' strings to native clients.
    No fallback — unknown prefixes raise immediately.
    """
    _PREFIX_TO_PROVIDER = {
        "anthropic": "anthropic",
        "openai":    "openai",
        "google":    "google",
        "qwen":      "qwen",
        "x-ai":      "xai",
        "vllm":      "vllm",
    }

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        google_key: Optional[str] = None,
        qwen_key: Optional[str] = None,
        xai_key: Optional[str] = None,
        vllm_key: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        timeout_s: int = 120,
    ):
        self._clients: Dict[str, BaseAPIClient] = {}
        if anthropic_key:
            self._clients["anthropic"] = AnthropicClient(anthropic_key, timeout_s)
        if openai_key:
            self._clients["openai"] = OpenAIClient(openai_key, timeout_s)
        if google_key:
            self._clients["google"] = GoogleClient(google_key, timeout_s)
        if qwen_key:
            self._clients["qwen"] = QwenClient(qwen_key, timeout_s)
        if xai_key:
            self._clients["xai"] = XAIClient(xai_key, timeout_s)
        # Registered on the base URL, not the key: a vLLM server started
        # without --api-key accepts any bearer token.
        if vllm_base_url:
            self._clients["vllm"] = VLLMClient(vllm_key, vllm_base_url, timeout_s)

    def provider_for(self, full_model_name: str) -> Optional[str]:
        """Provider name for a model string, or None if no prefix matches."""
        for prefix, provider in self._PREFIX_TO_PROVIDER.items():
            if full_model_name.startswith(f"{prefix}/"):
                return provider
        return None

    def resolve(self, full_model_name: str) -> Tuple[str, BaseAPIClient]:
        """Extracts provider to send API request to and model name separately from input provider/model_name"""
        for prefix, provider in self._PREFIX_TO_PROVIDER.items():
            if full_model_name.startswith(f"{prefix}/"):
                native_id = full_model_name[len(prefix) + 1:]
                if provider not in self._clients:
                    raise RuntimeError(
                        f"Model {full_model_name!r} needs provider {provider!r} "
                        f"but no API key was supplied."
                    )
                return native_id, self._clients[provider]
        raise RuntimeError(
            f"No native client registered for model {full_model_name!r}. "
            f"Known prefixes: {list(self._PREFIX_TO_PROVIDER)}"
        )

    def chat_completion(
        self,
        full_model_name: str,
        messages: List[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: Optional[str],
        hide_reasoning: bool,
    ) -> Tuple[str, dict]:
        native_id, client = self.resolve(full_model_name)
        return client.chat_completion(
            native_id,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            hide_reasoning=hide_reasoning,
        )


# ── Model Manager ──────────────────────────────────────────────────────────

class APIModelManager:

    def __init__(
        self,
        model_names: List[str],
        outputs_dir: str,
        router: RoutingClient,
        max_retries: int = 2,
        retry_backoff_s: float = 2.0,
        max_total_retries: int = 12,
        cache_path: Optional[str] = None,
        served_quantization: Optional[str] = None,
    ):
        self.model_names = model_names
        self.outputs_dir = outputs_dir
        self.router = router
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_total_retries = max_total_retries
        self.cache_path = cache_path
        self.served_quantization = served_quantization

        self.current_model_name: Optional[str] = None
        self.current_model_key: Optional[str] = None
        self.current_reasoning_effort: Optional[str] = None
        self.total_retries_this_model: int = 0
        self.abort_model: bool = False

        os.makedirs(self.outputs_dir, exist_ok=True)
        self._cache: Dict[str, dict] = {}
        self._load_cache()

    # ── response cache (resumability) ──────────────────────────────────────
    # Answers are keyed by (model key, condition, messages) and appended to a
    # JSONL sidecar, so re-running an interrupted model replays what was already
    # collected and only calls the API for the items still missing. A serving
    # session can disappear mid-run, and the reasoning condition is slow.
    #
    # It is also the provenance record, not just a convenience: an identical
    # reasoning request does not reproduce even greedily (see
    # GEN_PARAMS_REASONING_OVERRIDES), so a replayed answer is the one actually
    # collected. Keep the JSONL alongside the CSVs when archiving.

    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        with open(self.cache_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logging.warning("Skipping unreadable cache line in %s", self.cache_path)
                    continue
                self._cache[rec["key"]] = rec
        logging.info("Loaded %d cached responses from %s", len(self._cache), self.cache_path)

    def _cache_key(self, messages: List[dict]) -> str:
        key_parts = {
            "protocol":  CACHE_PROTOCOL_VERSION,
            "model_key": self.current_model_key,
            "model":     self.current_model_name,
            "effort":    self.current_reasoning_effort,
            "params":    self._gen_params(),
            "messages":  messages,
        }
        payload = json.dumps(key_parts, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_store(self, key: str, raw: str, meta: dict):
        rec = {"key": key, "raw": raw, "meta": meta}
        self._cache[key] = rec
        if not self.cache_path:
            return
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── per-call parameters ────────────────────────────────────────────────

    def _gen_params(self) -> dict:
        """Generation parameters for the current model and condition."""
        if not self.current_reasoning_effort:
            return GEN_PARAMS
        provider = self.router.provider_for(self.current_model_name or "")
        return GEN_PARAMS_REASONING_OVERRIDES.get(provider, GEN_PARAMS_REASONING)

    def _serving_notes(self) -> dict:
        """Serving details the API does not return (dtype/quantization)."""
        notes: dict = {}
        for pattern, values in MODEL_SERVING_NOTES.items():
            if pattern in (self.current_model_name or ""):
                notes.update(values)
        if self.served_quantization:
            notes["served_quantization"] = self.served_quantization
        return notes

    # ── task loading ───────────────────────────────────────────────────────

    def load_task_module(self, task_path: str):
        spec = importlib.util.spec_from_file_location("task_module", task_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def discover_tasks(self, task_dir: str) -> List[str]:
        p = Path(task_dir)
        if not p.exists():
            logging.error("Task directory does not exist: %s", task_dir)
            return []
        files = sorted(str(f) for f in p.glob("*.py") if not f.name.startswith("__"))
        logging.info("Discovered %d task files in %s", len(files), task_dir)
        return files

    # ── answer extraction ──────────────────────────────────────────────────

    def _extract_answer(self, raw_text: str, valid_answers: List[str], answer_pattern) -> str:
        text = (raw_text or "").strip()
        if text in valid_answers:
            return text
        if answer_pattern is not None:
            m = answer_pattern.search(text)
            if m and m.group(1) in valid_answers:
                return m.group(1)
        extracted = base_functions._extract_valid_answer_from_text(text, valid_answers)
        if extracted is not None:
            return extracted
        logging.warning("Could not parse valid answer from: %r", text)
        return ""

    def _answer_from_raw(self, raw: str, meta: dict, valid_answers: List[str],
                         answer_pattern) -> str:
        """Extract the answer, ignoring any inline chain of thought.

        `raw` is kept intact in the output CSV; only the text used for matching
        is narrowed to what follows the final </think>.
        """
        visible, reasoning = _split_inline_reasoning(raw)
        if reasoning is not None:
            meta["inline_reasoning_stripped"] = True
            meta.setdefault("thinking_content", reasoning)

        if meta.get("finish_reason") in TRUNCATED_FINISH_REASONS:
            # The model was cut off mid-generation. Trust the output only if it
            # had already closed its reasoning, or if what came back is itself
            # exactly a valid answer; otherwise matching against a partial
            # response would invent an answer the model never gave.
            if reasoning is None and visible.strip() not in valid_answers:
                meta["truncated_no_answer"] = True
                logging.warning(
                    "Truncated before an answer was produced | model %r | condition %r "
                    "| recording missing. Partial output: %r",
                    self.current_model_name, self.current_reasoning_effort, visible[:120],
                )
                return ""
        return self._extract_answer(visible, valid_answers, answer_pattern)

    def _build_probs_dict(self, raw: str, meta: dict) -> dict:
        """Per-call record. Every key here becomes a column in the output CSV."""
        probs_dict = {
            "raw_answer": raw,
            "requested_model": self.current_model_name,
            "condition": next(
                (s for s, e in REASONING_CONDITIONS if e == self.current_reasoning_effort),
                UNKNOWN_CONDITION_LABEL,
            ),
        }
        probs_dict.update(meta)
        probs_dict.update(self._serving_notes())
        return probs_dict

    # ── generation (injected into base_functions) ──────────────────────────

    def _api_generate_answer_and_logits(
        self, model, tokenizer, outlines_model, messages,
        valid_answers, model_name, ANSWER_PATTERN=None,
    ) -> Tuple[str, dict, dict]:
        del model, tokenizer, outlines_model, model_name

        last_err = None
        raw, meta = "", {}

        cache_key = self._cache_key(messages)
        cached = self._cache.get(cache_key)
        if cached is not None and cached.get("meta", {}).get("finish_reason") in TRUNCATED_FINISH_REASONS:
            # A truncated response carries no usable answer; re-request it rather
            # than replaying a dead cell on every resume.
            cached = None
        if cached is not None:
            meta = dict(cached["meta"])
            meta["from_cache"] = True
            answer = self._answer_from_raw(
                cached["raw"], meta, valid_answers, ANSWER_PATTERN
            )
            return answer, {}, self._build_probs_dict(cached["raw"], meta)

        for attempt in range(1, self.max_retries + 1):
            try:
                params = self._gen_params()
                raw, meta = self.router.chat_completion(
                    self.current_model_name,
                    messages,
                    reasoning_effort=self.current_reasoning_effort,
                    **params,
                )
 
                if meta.get("finish_reason") in ("length", "MAX_TOKENS", "max_tokens"):
                    logging.warning(
                        "Truncated output | model %r | condition %r | finish_reason %r",
                        self.current_model_name, self.current_reasoning_effort, meta["finish_reason"],
                    )

                # Per-model rate limit throttle
                delay = next(
                    (d for pattern, d in MODEL_REQUEST_DELAY_S.items()
                    if pattern in self.current_model_name),
                    0.0,
                )
                if delay > 0:
                    time.sleep(delay)
                break
            except error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    pass
                msg = f"HTTP {e.code}: {body[:800]}"
                if e.code in (400, 401, 402, 403, 404):
                    self.abort_model = True
                    raise RuntimeError(msg)
                last_err = RuntimeError(msg)
                if e.code == 429:
                    logging.warning("Rate limited (429). Waiting 60s before retry.")
                    time.sleep(60.0)
                    last_err = RuntimeError(msg)
                    continue
            except Exception as exc:
                last_err = exc

            self.total_retries_this_model += 1
            if self.total_retries_this_model >= self.max_total_retries:
                self.abort_model = True
                raise RuntimeError(
                    f"Aborting after {self.total_retries_this_model} retries: {last_err}"
                )

            sleep_s = self.retry_backoff_s * attempt
            logging.warning(
                "API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt, self.max_retries, last_err, sleep_s,
            )
            time.sleep(sleep_s)

        if last_err is not None and not raw:
            logging.error("API generation failed. Last error: %s", last_err)
            # Deliberately not cached: a failed cell must be retried on resume.
            return "ERROR", {}, self._build_probs_dict("ERROR", meta)

        self._cache_store(cache_key, raw, meta)
        answer = self._answer_from_raw(raw, meta, valid_answers, ANSWER_PATTERN)
        return answer, {}, self._build_probs_dict(raw, meta)

    # ── task runner ────────────────────────────────────────────────────────

    def run_task(self, task_module, task_name: str, context_modes=("no_context", "with_context")) -> Optional[pd.DataFrame]:
        # Report the condition suffix (nr/r), not the effort label: the two come
        # apart wherever VLLM_REASONING_EFFORT overrides the panel-wide value,
        # and printing "condition 'low'" for a run that transmits max is
        # needlessly confusing.
        condition = next(
            (s for s, e in REASONING_CONDITIONS if e == self.current_reasoning_effort),
            UNKNOWN_CONDITION_LABEL,
        )
        logging.info(
            "Running task %r | model %r | condition %r",
            task_name, self.current_model_key, condition,
        )
        orig_bf = base_functions.generate_answer_and_logits
        orig_tm = getattr(task_module, "generate_answer_and_logits", None)

        base_functions.generate_answer_and_logits = self._api_generate_answer_and_logits
        if orig_tm is not None:
            task_module.generate_answer_and_logits = self._api_generate_answer_and_logits

        try:
            result_df = task_module.run_task(
                model=None, tokenizer=None, outlines_model=None,
                model_key=self.current_model_key,
                context_modes=context_modes,
                
            )
        finally:
            base_functions.generate_answer_and_logits = orig_bf
            if orig_tm is not None:
                task_module.generate_answer_and_logits = orig_tm

        if result_df is not None and not result_df.empty:
            fname = f"{self.current_model_key}_{task_name}_results.csv"
            out = os.path.join(self.outputs_dir, fname)
            result_df.to_csv(out, index=False)
            logging.info("Saved %s", out)
            gc.collect()
            return result_df

        logging.warning("Task %r returned empty results.", task_name)
        return None

    # ── main loop ──────────────────────────────────────────────────────────

    def run_all(self, task_paths: List[str]):
        task_modules = {}
        for tp in task_paths:
            name = os.path.splitext(os.path.basename(tp))[0]
            try:
                task_modules[name] = self.load_task_module(tp)
            except Exception as e:
                logging.error("Failed to load %r: %s", tp, e)

        if not task_modules:
            logging.error("No task modules loaded.")
            return

        for model_name in self.model_names:
            # Determine which conditions to run for this model
            active_conditions = REASONING_CONDITIONS
            for pattern, suffixes in MODEL_CONDITION_OVERRIDES.items():
                if pattern in model_name:
                    active_conditions = [(s, e) for s, e in REASONING_CONDITIONS if s in suffixes]
                    break
            # loop over all reasoning conditions, build dataset for each condition
            for suffix, effort in active_conditions:
                self.current_model_name    = model_name
                self.current_reasoning_effort = effort
                # e.g. anthropic__claude-opus-4-6__nr
                self.current_model_key    = f"{_safe_model_key(model_name)}__{suffix}"
                self.total_retries_this_model = 0
                self.abort_model          = False

                logging.info("\n=== %s | %s ===", self.current_model_name, suffix)

                for task_name, task_module in task_modules.items():
                    try:
                        # This script runs no_context only — with_context intentionally excluded.
                        self.run_task(task_module, task_name, context_modes=("no_context",))
                    except Exception as e:
                        logging.error(
                            "Error | task %r | model %r | condition %r: %s",
                            task_name, self.current_model_key, suffix, e,
                        )
                    if self.abort_model:
                        logging.error("Aborting %r.", self.current_model_key)
                        break


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",       default="")
    p.add_argument("--models-file",  default="")
    p.add_argument("--task-dir",     default="tasks")
    p.add_argument("--outputs-dir",  default="outputs_api")
    p.add_argument("--task-filter",  default="")
    p.add_argument("--timeout",      type=int,   default=180)
    p.add_argument("--max-retries",  type=int,   default=2)
    p.add_argument("--max-total-retries", type=int, default=6)
    p.add_argument("--retry-backoff", type=float, default=2.0)
    
    # API key env-var names (override if needed)
    p.add_argument("--anthropic-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--openai-key-env",    default="OPENAI_API_KEY")
    p.add_argument("--google-key-env",    default="GOOGLE_API_KEY")
    p.add_argument("--qwen-key-env",      default="DASHSCOPE_API_KEY")
    p.add_argument("--xai-key-env",       default="XAI_API_KEY")
    p.add_argument("--vllm-key-env",      default="VLLM_API_KEY")

    # Self-hosted vLLM: the address changes per serving session, so it is a
    # run-time argument. Falls back to $VLLM_BASE_URL.
    p.add_argument("--vllm-base-url", default="",
                   help="Base URL of the vLLM OpenAI-compatible server, e.g. https://host/v1")
    p.add_argument("--served-quantization", default="",
                   help="Served dtype/quantization for the models in this run (e.g. bf16, fp8). "
                        "No API reports this, so it is recorded from the command line.")
    p.add_argument("--resume-cache", default="",
                   help="Path to the response cache. Defaults to <outputs-dir>/response_cache.jsonl")
    p.add_argument("--no-resume-cache", action="store_true",
                   help="Disable the response cache (every item is re-requested).")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the vLLM reachability/model check before the run.")
    return p.parse_args()


def main():
    args = parse_args()

    def _load_models_file(path: str) -> List[str]:
        with open(path, encoding="utf-8") as f:
            return [
                l.strip() for l in f
                if l.strip() and not l.strip().startswith("#")
            ]

    model_names: List[str] = []
    if args.models_file:
        model_names = _load_models_file(args.models_file)
    if not model_names:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise RuntimeError("No models provided.")

    vllm_base_url = args.vllm_base_url or os.environ.get("VLLM_BASE_URL", "")
    vllm_models = [m for m in model_names if m.startswith("vllm/")]
    if vllm_models and not vllm_base_url:
        raise RuntimeError(
            "vllm/ models requested but no server address given. "
            "Pass --vllm-base-url or set VLLM_BASE_URL."
        )
    if vllm_models and not args.skip_preflight:
        preflight_vllm(
            vllm_base_url,
            os.environ.get(args.vllm_key_env),
            args.timeout,
            vllm_models,
        )

    router = RoutingClient(
        anthropic_key=os.environ.get(args.anthropic_key_env),
        openai_key=os.environ.get(args.openai_key_env),
        google_key=os.environ.get(args.google_key_env),
        qwen_key=os.environ.get(args.qwen_key_env),
        xai_key=os.environ.get(args.xai_key_env),
        vllm_key=os.environ.get(args.vllm_key_env),
        vllm_base_url=vllm_base_url,
        timeout_s=args.timeout,
    )

    cache_path = None
    if not args.no_resume_cache:
        cache_path = args.resume_cache or os.path.join(
            args.outputs_dir, "response_cache.jsonl"
        )
        os.makedirs(args.outputs_dir, exist_ok=True)

    manager = APIModelManager(
        model_names=model_names,
        outputs_dir=args.outputs_dir,
        router=router,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff,
        max_total_retries=args.max_total_retries,
        cache_path=cache_path,
        served_quantization=args.served_quantization or None,
    )

    task_paths = manager.discover_tasks(args.task_dir)
    if args.task_filter.strip():
        selected = {t.strip() for t in args.task_filter.split(",")}
        task_paths = [p for p in task_paths
                      if os.path.splitext(os.path.basename(p))[0] in selected]

    manager.run_all(task_paths)


if __name__ == "__main__":
    main()