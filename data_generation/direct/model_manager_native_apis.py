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

Reads the task definitions in <task-dir>/ (+ their tasks/jsonl_data/*.jsonl items).
Writes <outputs-dir>/<safe-model>__<condition>_<task>_results.csv and a run log,
under outputs_api/.

Usage: python model_manager_native_apis.py --models-file api_models.txt
"""

import abc
import argparse
import gc
import importlib.util
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error, request

import pandas as pd
import base_functions


# ── Logging ────────────────────────────────────────────────────────────────

os.makedirs("outputs_api", exist_ok=True)
log_file = os.path.join(
    "outputs_api", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
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

# ── Utilities ──────────────────────────────────────────────────────────────

def _safe_model_key(model_name: str) -> str:
    key = model_name.strip().replace("/", "__").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


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
        return json.loads(resp.read().decode("utf-8")), ts, safe_url


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
    }

    def __init__(
        self,
        anthropic_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        google_key: Optional[str] = None,
        qwen_key: Optional[str] = None,
        xai_key: Optional[str] = None,
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
    ):
        self.model_names = model_names
        self.outputs_dir = outputs_dir
        self.router = router
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_total_retries = max_total_retries

        self.current_model_name: Optional[str] = None
        self.current_model_key: Optional[str] = None
        self.current_reasoning_effort: Optional[str] = None
        self.total_retries_this_model: int = 0
        self.abort_model: bool = False

        os.makedirs(self.outputs_dir, exist_ok=True)

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

    # ── generation (injected into base_functions) ──────────────────────────

    def _api_generate_answer_and_logits(
        self, model, tokenizer, outlines_model, messages,
        valid_answers, model_name, ANSWER_PATTERN=None,
    ) -> Tuple[str, dict, dict]:
        del model, tokenizer, outlines_model, model_name

        last_err = None
        raw, meta = "", {}

        for attempt in range(1, self.max_retries + 1):
            try:
                params = GEN_PARAMS_REASONING if self.current_reasoning_effort else GEN_PARAMS
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
            probs_dict = {
                "raw_answer": "ERROR",
                "requested_model": self.current_model_name,
                "condition": next(
                    (s for s, e in REASONING_CONDITIONS if e == self.current_reasoning_effort),
                    UNKNOWN_CONDITION_LABEL,
                ),
            }
            probs_dict.update(meta)
            return "ERROR", {}, probs_dict

        answer = self._extract_answer(raw, valid_answers, ANSWER_PATTERN)
        probs_dict = {
            "raw_answer": raw,
            "requested_model": self.current_model_name,
            "condition": next(
                (s for s, e in REASONING_CONDITIONS if e == self.current_reasoning_effort),
                UNKNOWN_CONDITION_LABEL,
            ),
        }
        probs_dict.update(meta)
        return answer, {}, probs_dict

    # ── task runner ────────────────────────────────────────────────────────

    def run_task(self, task_module, task_name: str, context_modes=("no_context", "with_context")) -> Optional[pd.DataFrame]:
        logging.info(
            "Running task %r | model %r | condition %r",
            task_name, self.current_model_key, self.current_reasoning_effort,
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

    router = RoutingClient(
        anthropic_key=os.environ.get(args.anthropic_key_env),
        openai_key=os.environ.get(args.openai_key_env),
        google_key=os.environ.get(args.google_key_env),
        qwen_key=os.environ.get(args.qwen_key_env),
        xai_key=os.environ.get(args.xai_key_env),
        timeout_s=args.timeout,
    )


    manager = APIModelManager(
        model_names=model_names,
        outputs_dir=args.outputs_dir,
        router=router,
        max_retries=args.max_retries,
        retry_backoff_s=args.retry_backoff,
        max_total_retries=args.max_total_retries,
    )

    task_paths = manager.discover_tasks(args.task_dir)
    if args.task_filter.strip():
        selected = {t.strip() for t in args.task_filter.split(",")}
        task_paths = [p for p in task_paths
                      if os.path.splitext(os.path.basename(p))[0] in selected]

    manager.run_all(task_paths)


if __name__ == "__main__":
    main()