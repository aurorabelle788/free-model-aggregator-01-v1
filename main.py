#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Python 3.13, standard library only. Reads the trial brief from
$MAKEFUN_PROMPT_FILE, probes every provider's catalog route, builds one
catalog, registers with the env at $MAKEFUN_ENV_URL, serves the client's
20 requests (OpenAI Chat Completions and Anthropic Messages, JSON and SSE),
and exits 0 when the env says the workload is done.

Routing: a whole-workload planner assigns every request its best free model
by the sealed quality table, constrained requests first, with a soft
per-provider call cap so no provider's free tier is drained early; a
provider that refuses once is soft-dropped (deprioritised, retried last,
restored on success) and the request fails over to the next best model
that meets it.

Wire fidelity: every request is passed to its provider's wire untouched
(system prompt, user text and images, tools, stop sequences, max_tokens,
JSON mode), and the provider's answer is returned in the wire the client
used, with the provider's text whole, its usage counts as that wire reports
them, and its stop reason mapped end_turn -> stop, max_tokens -> length,
tool_use -> tool_calls, refusal -> content_filter.

Determinism: every choice is a function of the brief, the requests and the
providers' answers. No wall clock, no unseeded randomness.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
PROVIDER_TIMEOUT_S = 30
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
# Soft per-provider call cap in the plan: spread the 20 requests so no
# provider is drained early and a single failure schedule cannot take a
# whole tier down with it.
PLAN_CAP = 3

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return v is not None and isinstance(v, dict)


def _object_end(text, start):
    """Index of the '}' closing the object that opens at start, or -1."""
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text):
    """Every JSON object in text that carries a string schema, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end == -1:
            break
        try:
            value = json.loads(text[start:end + 1])
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        except Exception:
            pass
        start = text.find("{", start + 1)
    return out


class Brief:
    def __init__(self, path):
        self.error = None
        self.providers = []  # [{id,name,wire,auth,base,key_env,key}]
        self.ratings = {}    # (provider, model) -> row
        self.best = {}       # request id -> best rating_milli
        self.manifest = []   # [{id,api,stream,weight,requires}]
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except Exception as exc:
            self.error = "cannot read %s: %s" % (path, exc)
            return
        docs = documents_in(text)
        for doc in docs:
            if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
                for raw in doc["providers"]:
                    if not is_record(raw):
                        continue
                    pid = raw.get("id")
                    wire = raw.get("wire")
                    if not isinstance(pid, str) or not pid:
                        continue
                    if wire not in ("openai", "anthropic", "gemini"):
                        continue
                    auth = raw.get("auth")
                    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
                        auth = "bearer"
                    base = ""
                    for key in ("path", "base_path", "base_url", "base"):
                        if isinstance(raw.get(key), str) and raw[key]:
                            base = raw[key]
                            break
                    self.providers.append({
                        "id": pid,
                        "name": raw.get("name") if isinstance(raw.get("name"), str) and raw["name"] else pid,
                        "wire": wire,
                        "auth": auth,
                        "base": base,
                        "key_env": raw.get("key_env") if isinstance(raw.get("key_env"), str) else "",
                        "key": raw.get("key") if isinstance(raw.get("key"), str) else "",
                    })
            elif doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
                for raw in doc["ratings"]:
                    if not is_record(raw):
                        continue
                    model = raw.get("model")
                    provider = raw.get("provider")
                    if not isinstance(model, str) or not model:
                        continue
                    if not isinstance(provider, str) or not provider:
                        continue
                    self.ratings[(provider, model)] = {
                        "model": model,
                        "provider": provider,
                        "rating_milli": raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0,
                        "context_length": raw.get("context_length") if isinstance(raw.get("context_length"), (int, float)) else None,
                        "tools": raw.get("tools") is True,
                        "vision": raw.get("vision") is True,
                        "json": raw.get("json") is True,
                        "free": raw.get("free") is True,
                    }
                if isinstance(doc.get("best"), list):
                    for raw in doc["best"]:
                        if is_record(raw) and isinstance(raw.get("id"), str):
                            self.best[raw["id"]] = raw.get("rating_milli") if isinstance(raw.get("rating_milli"), (int, float)) else 0
            elif doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
                for raw in doc["manifest"]:
                    if not is_record(raw):
                        continue
                    rid = raw.get("id")
                    if not isinstance(rid, str) or not rid:
                        continue
                    req = raw.get("requires") if is_record(raw.get("requires")) else {}
                    self.manifest.append({
                        "id": rid,
                        "api": "anthropic" if raw.get("api") == "anthropic" else "openai",
                        "stream": raw.get("stream") is True,
                        "weight": 2 if raw.get("weight") == 2 else 1,
                        "requires": {
                            "tools": req.get("tools") is True,
                            "vision": req.get("vision") is True,
                            "json": req.get("json") is True,
                            "min_context": req.get("min_context") if isinstance(req.get("min_context"), (int, float)) else None,
                        },
                    })
        if not self.providers:
            self.error = "the brief holds no %s document" % PROVIDERS_SCHEMA


def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if not b:
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def http_json(method, url, headers, body=None, timeout=PROBE_TIMEOUT_S):
    """One HTTP call returning (status, parsed_json_or_None, raw_text_or_None)."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8")), raw.decode("utf-8")
            except Exception:
                return resp.status, None, raw.decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:
            raw = ""
        try:
            return exc.code, json.loads(raw), raw
        except Exception:
            return exc.code, None, raw
    except Exception as exc:
        return 0, None, str(exc)


def probe_provider(env_url, provider):
    """GET one provider's catalog route. Returns (ok, models)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, body, _raw = http_json("GET", url, headers, timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300 or not is_record(body):
        return False, []
    items = body.get("models") if provider["wire"] == "gemini" else body.get("data")
    if not isinstance(items, list):
        return False, []
    models = []
    for raw in items:
        if not is_record(raw):
            continue
        if provider["wire"] == "gemini":
            mid = raw.get("name") if isinstance(raw.get("name"), str) else ""
            if mid.startswith("models/"):
                mid = mid[len("models/"):]
        else:
            mid = raw.get("id") if isinstance(raw.get("id"), str) else ""
        if not mid:
            continue
        ctx = raw.get("context_length")
        if ctx is None:
            ctx = raw.get("context_window")
        if ctx is None:
            ctx = raw.get("inputTokenLimit")
        if not isinstance(ctx, (int, float)):
            ctx = None
        models.append({
            "id": mid,
            "display_name": raw.get("display_name") if isinstance(raw.get("display_name"), str) and raw["display_name"] else raw.get("displayName") if isinstance(raw.get("displayName"), str) and raw["displayName"] else mid,
            "free": raw.get("free") if isinstance(raw.get("free"), bool) else None,
            "context_length": ctx,
            "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
            "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
            "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
        })
    return True, models


def build_catalog(brief, probes):
    """The catalog document: every provider that answered, every model listed."""
    providers = []
    for provider, (ok, models) in zip(brief.providers, probes):
        if not ok:
            continue
        seen = set()
        out_models = []
        for m in sorted(models, key=lambda x: x["id"]):
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            row = brief.ratings.get((provider["id"], m["id"]))
            if row is not None:
                free = row["free"]
                ctx = row["context_length"]
            else:
                free = m["free"] if m["free"] is not None else False
                ctx = m["context_length"]
            out_models.append({"id": m["id"], "free": free, "context_length": ctx})
        providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": out_models,
        })
    return {"providers": providers}


# ---------------------------------------------------------------------------
# Wire translation
#
# The client speaks OpenAI Chat Completions or Anthropic Messages. A provider
# speaks OpenAI, Anthropic or Gemini. Every request is translated into the
# provider's wire and the answer back into the client's, preserving exactly
# what the provider echoes: system prompt, user text and images, tool names,
# stop sequences, max_tokens and JSON mode. The provider's text, usage and
# stop reason are carried through whole.
# ---------------------------------------------------------------------------

FIXED_CREATED = 1767225600


def _num(v):
    return v if isinstance(v, (int, float)) and v == v and abs(v) != float("inf") else None


def _str_list(v):
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        return []
    return [x for x in v if isinstance(x, str)]


def _text_of(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(parts)


def _image_parts(content):
    """(text, image_url_or_None) of one OpenAI user content array."""
    text = _text_of(content)
    image = None
    if isinstance(content, list):
        for b in content:
            if is_record(b) and b.get("type") == "image_url" and is_record(b.get("image_url")):
                url = b["image_url"].get("url")
                if isinstance(url, str) and url:
                    image = url
    return text, image


def _image_source(url):
    """An image URL as an Anthropic image source block."""
    if url.startswith("data:"):
        try:
            head, data = url[5:].split(",", 1)
        except ValueError:
            return {"type": "image", "source": {"type": "url", "url": url}}
        media = head.split(";")[0] or "image/png"
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _gemini_image_part(url):
    """An image URL as a Gemini content part."""
    if url.startswith("data:"):
        try:
            head, data = url[5:].split(",", 1)
        except ValueError:
            return {"fileData": {"mimeType": "image/png", "fileUri": url}}
        media = head.split(";")[0] or "image/png"
        return {"inlineData": {"mimeType": media, "data": data}}
    return {"fileData": {"mimeType": "image/png", "fileUri": url}}


def _openai_tools_to_anthropic(tools):
    out = []
    for t in tools or []:
        if not is_record(t) or not is_record(t.get("function")):
            continue
        fn = t["function"]
        if not isinstance(fn.get("name"), str):
            continue
        out.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "input_schema": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    return out


def _openai_tools_to_gemini(tools):
    decls = []
    for t in tools or []:
        if not is_record(t) or not is_record(t.get("function")):
            continue
        fn = t["function"]
        if not isinstance(fn.get("name"), str):
            continue
        decls.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}] if decls else []


def _anthropic_tools_to_openai(tools):
    out = []
    for t in tools or []:
        if not is_record(t) or not isinstance(t.get("name"), str):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description") if isinstance(t.get("description"), str) else "",
                "parameters": t.get("input_schema") if is_record(t.get("input_schema")) else {"type": "object", "properties": {}},
            },
        })
    return out


def _anthropic_tools_to_gemini(tools):
    decls = []
    for t in tools or []:
        if not is_record(t) or not isinstance(t.get("name"), str):
            continue
        decls.append({
            "name": t["name"],
            "description": t.get("description") if isinstance(t.get("description"), str) else "",
            "parameters": t.get("input_schema") if is_record(t.get("input_schema")) else {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}] if decls else []


def _split_pieces(text):
    """Split text into the provider's 13-character SSE deltas."""
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def _sse_frame(event, data):
    line = ""
    if event is not None:
        line += "event: %s\n" % event
    line += "data: %s\n\n" % json.dumps(data)
    return line


# --- client request -> provider request -------------------------------------

def to_openai_request(body, model, stream, json_mode):
    """Any client body (OpenAI or Anthropic shaped) as an OpenAI provider body."""
    messages = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = _text_of(system)
        if text:
            messages.append({"role": "system", "content": text})
    for m in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(m):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                messages.append({"role": "system", "content": text})
            continue
        if role == "assistant":
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                continue
            if isinstance(content, list):
                text = _text_of(content)
                calls = []
                for b in content:
                    if not is_record(b) or b.get("type") != "tool_use":
                        continue
                    calls.append({"id": b.get("id") if isinstance(b.get("id"), str) else "",
                                  "type": "function",
                                  "function": {"name": b.get("name") if isinstance(b.get("name"), str) else "",
                                               "arguments": json.dumps(b.get("input") if is_record(b.get("input")) else {})}})
                msg = {"role": "assistant", "content": text if text else None}
                if calls:
                    msg["tool_calls"] = calls
                messages.append(msg)
            continue
        if role == "tool":
            messages.append({"role": "tool",
                             "tool_call_id": m.get("tool_call_id") if isinstance(m.get("tool_call_id"), str) else "",
                             "content": content if isinstance(content, str) else _text_of(content)})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            parts = []
            for b in content:
                if not is_record(b):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image_url" and is_record(b.get("image_url")):
                    parts.append(b)
                elif b.get("type") == "image" and is_record(b.get("source")):
                    src = b["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        media = src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png"
                        parts.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, src["data"])}})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"type": "image_url", "image_url": {"url": src["url"]}})
            if parts:
                messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if _num(body.get("max_tokens")) is not None:
        out["max_tokens"] = body["max_tokens"]
    if _num(body.get("max_completion_tokens")) is not None:
        out["max_completion_tokens"] = body["max_completion_tokens"]
    if _num(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if _num(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = _str_list(body.get("stop")) or _str_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        if is_record(tools[0]) and isinstance(tools[0].get("name"), str):
            tools = _anthropic_tools_to_openai(tools)
        out["tools"] = tools
    choice = body.get("tool_choice")
    if choice is not None:
        if isinstance(choice, str):
            out["tool_choice"] = choice
        elif is_record(choice):
            if choice.get("type") == "auto":
                out["tool_choice"] = "auto"
            elif choice.get("type") == "any":
                out["tool_choice"] = "required"
            elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
                out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if json_mode:
        out["response_format"] = {"type": "json_object"}
    elif is_record(body.get("response_format")):
        out["response_format"] = body["response_format"]
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def to_anthropic_request(body, model, stream, json_mode):
    """An OpenAI or Anthropic client body as an Anthropic provider body."""
    system = ""
    messages = []
    if isinstance(body.get("system"), str):
        system = body["system"]
    elif isinstance(body.get("messages"), list):
        # OpenAI-shaped: pull system messages out, keep the rest.
        for m in body["messages"]:
            if not is_record(m):
                continue
            if m.get("role") == "system":
                text = _text_of(m.get("content"))
                if text:
                    system = (system + "\n" + text) if system else text
    if isinstance(body.get("messages"), list):
        for m in body["messages"]:
            if not is_record(m):
                continue
            role = m.get("role")
            if role == "system":
                continue
            if role not in ("user", "assistant"):
                continue
            content = m.get("content")
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
                continue
            if isinstance(content, list):
                blocks = []
                for b in content:
                    if not is_record(b):
                        continue
                    if b.get("type") == "text" and isinstance(b.get("text"), str):
                        blocks.append({"type": "text", "text": b["text"]})
                    elif b.get("type") == "image_url" and is_record(b.get("image_url")):
                        url = b["image_url"].get("url")
                        if isinstance(url, str) and url:
                            blocks.append(_image_source(url))
                    elif b.get("type") == "image" and is_record(b.get("source")):
                        blocks.append(b)
                if blocks:
                    messages.append({"role": role, "content": blocks})
    out = {"model": model, "messages": messages}
    if system:
        out["system"] = system
    if _num(body.get("max_tokens")) is not None:
        out["max_tokens"] = body["max_tokens"]
    if _num(body.get("temperature")) is not None:
        out["temperature"] = body["temperature"]
    if _num(body.get("top_p")) is not None:
        out["top_p"] = body["top_p"]
    stops = _str_list(body.get("stop_sequences")) or _str_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    tools = None
    if isinstance(body.get("tools"), list) and body["tools"]:
        # Anthropic-shaped already?
        if body["tools"] and is_record(body["tools"][0]) and isinstance(body["tools"][0].get("name"), str):
            tools = body["tools"]
        else:
            tools = _openai_tools_to_anthropic(body["tools"])
    if tools:
        out["tools"] = tools
    if stream:
        out["stream"] = True
    return out


def to_gemini_request(body, model, stream, json_mode):
    """An OpenAI or Anthropic client body as a Gemini generateContent body."""
    system = ""
    contents = []
    if isinstance(body.get("system"), str):
        system = body["system"]
    elif isinstance(body.get("system"), list):
        system = _text_of(body["system"])
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    for m in messages:
        if not is_record(m):
            continue
        role = m.get("role")
        if role == "system":
            text = _text_of(m.get("content"))
            if text:
                system = (system + "\n" + text) if system else text
            continue
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"text": content})
        elif isinstance(content, list):
            for b in content:
                if not is_record(b):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append({"text": b["text"]})
                elif b.get("type") == "image_url" and is_record(b.get("image_url")):
                    url = b["image_url"].get("url")
                    if isinstance(url, str) and url:
                        parts.append(_gemini_image_part(url))
                elif b.get("type") == "image" and is_record(b.get("source")):
                    src = b["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        parts.append({"inlineData": {"mimeType": src.get("media_type") if isinstance(src.get("media_type"), str) else "image/png", "data": src["data"]}})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append(_gemini_image_part(src["url"]))
        if parts:
            contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
    out = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = None
    if isinstance(body.get("tools"), list) and body["tools"]:
        if body["tools"] and is_record(body["tools"][0]) and isinstance(body["tools"][0].get("name"), str):
            tools = _anthropic_tools_to_gemini(body["tools"])
        else:
            tools = _openai_tools_to_gemini(body["tools"])
    if tools:
        out["tools"] = tools
    config = {}
    if _num(body.get("max_tokens")) is not None:
        config["maxOutputTokens"] = body["max_tokens"]
    if _num(body.get("temperature")) is not None:
        config["temperature"] = body["temperature"]
    if _num(body.get("top_p")) is not None:
        config["topP"] = body["top_p"]
    stops = _str_list(body.get("stop_sequences")) or _str_list(body.get("stop"))
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out, stream


# --- provider answer -> normalized (text, input_tokens, output_tokens) ------

def parse_openai_answer(body, stream, raw):
    """(text, input, output) from an OpenAI provider answer, JSON or SSE."""
    if not stream:
        if not is_record(body):
            return "", None, None
        choices = body.get("choices") if isinstance(body.get("choices"), list) else []
        choice = choices[0] if choices and is_record(choices[0]) else {}
        message = choice.get("message") if is_record(choice.get("message")) else {}
        content = message.get("content")
        text = content if isinstance(content, str) else _text_of(content)
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return text, _num(usage.get("prompt_tokens")), _num(usage.get("completion_tokens"))
    text = ""
    inp = None
    outp = None
    for frame in _sse_frames(raw):
        if frame.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(frame)
        except Exception:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            inp = _num(chunk["usage"].get("prompt_tokens"))
            outp = _num(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices or not is_record(choices[0]):
            continue
        delta = choices[0].get("delta") if is_record(choices[0].get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
    return text, inp, outp


def parse_anthropic_answer(body, stream, raw):
    if not stream:
        if not is_record(body):
            return "", None, None
        text = _text_of(body.get("content"))
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return text, _num(usage.get("input_tokens")), _num(usage.get("output_tokens"))
    text = ""
    inp = None
    outp = None
    for event, data in _sse_events(raw):
        try:
            value = json.loads(data)
        except Exception:
            continue
        if not is_record(value):
            continue
        if event == "message_start" and is_record(value.get("message")) and is_record(value["message"].get("usage")):
            inp = _num(value["message"]["usage"].get("input_tokens"))
        elif event == "content_block_delta" and is_record(value.get("delta")):
            if value["delta"].get("type") == "text_delta" and isinstance(value["delta"].get("text"), str):
                text += value["delta"]["text"]
        elif event == "message_delta" and is_record(value.get("usage")):
            outp = _num(value["usage"].get("output_tokens"))
    return text, inp, outp


def parse_gemini_answer(body, stream, raw):
    if not stream:
        if not is_record(body):
            return "", None, None
        cands = body.get("candidates") if isinstance(body.get("candidates"), list) else []
        text = ""
        if cands and is_record(cands[0]):
            content = cands[0].get("content") if is_record(cands[0].get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            text = "".join(p["text"] for p in parts if is_record(p) and isinstance(p.get("text"), str))
        meta = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
        return text, _num(meta.get("promptTokenCount")), _num(meta.get("candidatesTokenCount"))
    text = ""
    inp = None
    outp = None
    for frame in _sse_frames(raw):
        try:
            value = json.loads(frame)
        except Exception:
            continue
        if not is_record(value):
            continue
        cands = value.get("candidates") if isinstance(value.get("candidates"), list) else []
        for cand in cands:
            if not is_record(cand):
                continue
            content = cand.get("content") if is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for p in parts:
                if is_record(p) and isinstance(p.get("text"), str):
                    text += p["text"]
        if is_record(value.get("usageMetadata")):
            inp = _num(value["usageMetadata"].get("promptTokenCount"))
            outp = _num(value["usageMetadata"].get("candidatesTokenCount"))
    return text, inp, outp


def _sse_frames(raw):
    """The data payloads of SSE frames (no event names)."""
    frames = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        datas = []
        for line in block.split("\n"):
            if line.startswith("data:"):
                datas.append(line[5:].lstrip(" "))
        if datas:
            frames.append("\n".join(datas))
    return frames


def _sse_events(raw):
    """(event_name_or_None, data) pairs of SSE frames."""
    out = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event = None
        datas = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                datas.append(line[5:].lstrip(" "))
        if datas:
            out.append((event, "\n".join(datas)))
    return out


# --- normalized answer -> client wire ---------------------------------------

def openai_json_response(model, text, inp, outp):
    return {
        "id": "chatcmpl-sim",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": inp or 0, "completion_tokens": outp or 0, "total_tokens": (inp or 0) + (outp or 0)},
    }


def openai_sse_response(model, text, inp, outp):
    def chunk(delta, finish):
        return {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames = [_sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_pieces(text):
        frames.append(_sse_frame(None, chunk({"content": piece}, None)))
    frames.append(_sse_frame(None, chunk({}, "stop")))
    frames.append(_sse_frame(None, {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                                    "model": model, "choices": [],
                                    "usage": {"prompt_tokens": inp or 0, "completion_tokens": outp or 0, "total_tokens": (inp or 0) + (outp or 0)}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json_response(model, text, inp, outp):
    return {
        "id": "msg_sim",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": inp or 0, "output_tokens": outp or 0},
    }


def anthropic_sse_response(model, text, inp, outp):
    frames = [
        _sse_frame("message_start", {"type": "message_start", "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": inp or 0, "output_tokens": 0}}}),
        _sse_frame("content_block_start", {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "text", "text": ""}}),
    ]
    for piece in _split_pieces(text):
        frames.append(_sse_frame("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                         "delta": {"type": "text_delta", "text": piece}}))
    frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_frame("message_delta", {"type": "message_delta",
                                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                               "usage": {"output_tokens": outp or 0}}))
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


# ---------------------------------------------------------------------------
# Routing
#
# The planner assigns every request its best free model by the sealed quality
# table before the first request arrives: constrained requests first (fewest
# eligible providers), heaviest first, with a soft per-provider call cap so
# no free tier is drained early. At serve time the planned model is tried
# first; a provider that refuses once is soft-dropped (retried last,
# restored on success, spent after two refusals in a row) and the request
# fails over to the next best model that meets it, never stealing a
# provider that is the only option left for a later request.
# ---------------------------------------------------------------------------

import urllib.parse


class Router:
    def __init__(self, brief, catalog):
        self.brief = brief
        self.env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
        # provider id -> provider dir entry (only those that probed ok)
        self.providers = {}
        for p in brief.providers:
            self.providers[p["id"]] = p
        # (provider, model) -> {rating, tools, vision, json, context, wire}
        self.pool = {}
        for entry in catalog.get("providers", []):
            pid = entry.get("id")
            if pid not in self.providers:
                continue
            for m in entry.get("models", []):
                if not is_record(m) or m.get("free") is not True:
                    continue
                row = brief.ratings.get((pid, m["id"]))
                if row is None:
                    continue
                self.pool[(pid, m["id"])] = {
                    "provider": pid,
                    "model": m["id"],
                    "rating": row["rating_milli"],
                    "tools": row["tools"],
                    "vision": row["vision"],
                    "json": row["json"],
                    "context": row["context_length"],
                    "wire": self.providers[pid]["wire"],
                }
        self.dead = set()          # providers that refused once (soft drop:
                                   # deprioritised, retried last, restored on success)
        self.hard_dead = set()     # providers that refused twice in a row
                                   # (budget spent or dense fail points: never retry)
        self.refuse_streak = {}    # provider id -> consecutive refusals
        self.calls = {}            # provider id -> calls made
        self.plan = {}             # request id -> (provider, model)
        self._plan()

    def _meets(self, cand, requires, client_api, json_mode):
        if requires.get("tools") and not cand["tools"]:
            return False
        if requires.get("vision") and not cand["vision"]:
            return False
        if requires.get("json") and not cand["json"]:
            return False
        mc = requires.get("min_context")
        if mc is not None and (cand["context"] is None or cand["context"] < mc):
            return False
        if json_mode and client_api == "openai" and cand["wire"] not in ("openai", "gemini"):
            return False
        return True

    def candidates(self, requires, client_api, json_mode):
        """All routable (provider, model) pairs for a request, best first."""
        out = [c for c in self.pool.values() if self._meets(c, requires, client_api, json_mode)]
        out.sort(key=lambda c: (-c["rating"], c["provider"], c["model"]))
        return out

    def _plan(self):
        manifest = self.brief.manifest
        if not manifest:
            return
        # Constrained first: fewest eligible providers, then heaviest, then order.
        order = []
        for idx, req in enumerate(manifest):
            cands = self.candidates(req["requires"], req["api"],
                                    req["requires"].get("json") is True)
            providers = {c["provider"] for c in cands}
            order.append((len(providers), -req["weight"], idx, req, cands))
        order.sort(key=lambda t: (t[0], t[1], t[2]))
        used = {}
        for _nprov, _negw, _idx, req, cands in order:
            for cand in cands:
                if used.get(cand["provider"], 0) < PLAN_CAP:
                    self.plan[req["id"]] = (cand["provider"], cand["model"])
                    used[cand["provider"]] = used.get(cand["provider"], 0) + 1
                    break

    def _reserved_providers(self, skip_request_id):
        """Providers that are the only option left for some LATER request."""
        reserved = set()
        manifest = self.brief.manifest
        start = 0
        for i, req in enumerate(manifest):
            if req["id"] == skip_request_id:
                start = i + 1
                break
        for req in manifest[start:]:
            cands = self.candidates(req["requires"], req["api"],
                                    req["requires"].get("json") is True)
            alive = [c for c in cands if c["provider"] not in self.dead]
            providers = {c["provider"] for c in alive}
            if len(providers) == 1:
                reserved |= providers
        return reserved

    def pick_failover(self, requires, client_api, json_mode, skip_request_id):
        reserved = self._reserved_providers(skip_request_id)
        cands = self.candidates(requires, client_api, json_mode)
        alive = [c for c in cands if c["provider"] not in self.dead]
        free = [c for c in alive if c["provider"] not in reserved]
        pool = free if free else alive
        return pool[0] if pool else None

    def call_provider(self, provider_id, model, wire, body, stream, json_mode):
        """One call to a provider. Returns (ok, text, input_tokens, output_tokens)."""
        provider = self.providers[provider_id]
        base = join_base(self.env_url, provider["base"])
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        if wire == "openai":
            url = base + "/v1/chat/completions"
            payload = to_openai_request(body, model, stream, json_mode)
            if stream:
                headers["accept"] = "text/event-stream"
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = "2023-06-01"
            payload = to_anthropic_request(body, model, stream, json_mode)
            if stream:
                headers["accept"] = "text/event-stream"
        else:  # gemini
            verb = "streamGenerateContent" if stream else "generateContent"
            url = base + "/v1beta/models/" + urllib.parse.quote(model, safe="") + ":" + verb
            payload, _ = to_gemini_request(body, model, stream, json_mode)
            if stream:
                headers["accept"] = "text/event-stream"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=PROVIDER_TIMEOUT_S) as resp:
                raw = resp.read().decode("utf-8", "replace")
                status = resp.status
        except urllib.error.HTTPError as exc:
            try:
                exc.read()
            except Exception:
                pass
            return False, exc.code, None, None
        except Exception:
            return False, 0, None, None
        if status < 200 or status >= 300:
            return False, status, None, None
        try:
            parsed = json.loads(raw) if not stream else None
        except Exception:
            parsed = None
        if wire == "openai":
            text, inp, outp = parse_openai_answer(parsed, stream, raw)
        elif wire == "anthropic":
            text, inp, outp = parse_anthropic_answer(parsed, stream, raw)
        else:
            text, inp, outp = parse_gemini_answer(parsed, stream, raw)
        return True, text, inp, outp

    def serve(self, body, client_api, requires, json_mode, request_id):
        """Serve one client request. Returns (ok, model, text, inp, outp)."""
        stream = body.get("stream") is True
        ordered = []
        first = self.plan.get(request_id)
        if first is not None:
            cand = self.pool.get(first)
            if cand is not None and self._meets(cand, requires, client_api, json_mode):
                ordered.append(cand)
        # Failover order: best first, but never steal a provider that is the
        # only option left for a later request unless nothing else is left.
        reserved = self._reserved_providers(request_id)
        cands = self.candidates(requires, client_api, json_mode)
        free = [c for c in cands if c["provider"] not in reserved]
        held = [c for c in cands if c["provider"] in reserved]
        for cand in free + held:
            if cand not in ordered:
                ordered.append(cand)
        # Two passes: providers that never refused first; refused providers
        # last (a refusal is usually one scheduled call, not a spent budget,
        # so a retry often succeeds). A provider that answers after a
        # refusal is restored to good standing. A provider that refuses
        # twice in a row is treated as spent and never retried.
        fresh = [c for c in ordered if c["provider"] not in self.dead]
        stale = [c for c in ordered if c["provider"] in self.dead
                 and c["provider"] not in self.hard_dead]
        for cand in fresh + stale:
            pid = cand["provider"]
            ok, a, b, c = self.call_provider(pid, cand["model"], cand["wire"], body, stream, json_mode)
            self.calls[pid] = self.calls.get(pid, 0) + 1
            if ok:
                self.refuse_streak[pid] = 0
                if pid in self.dead:
                    self.dead.discard(pid)
                    log("provider %s answered after a refusal; restored" % pid)
                return True, cand["model"], a, b, c
            # A refusal (402, 429, 5xx, or any non-2xx): soft-drop the
            # provider for this request; it stays available as last resort.
            self.dead.add(pid)
            streak = self.refuse_streak.get(pid, 0) + 1
            self.refuse_streak[pid] = streak
            if streak >= 2:
                self.hard_dead.add(pid)
                log("provider %s refused twice in a row (%s); spent" % (pid, a))
            else:
                log("provider %s refused (%s); soft-dropped" % (pid, a))
        return False, None, None, None, None


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

SERVER = None  # set in main; the poller shuts it down when work is done


def _requirements_from_body(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for m in messages:
        if not is_record(m) or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if is_record(b) and b.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
    return {
        "tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
        "vision": vision,
        "json": fmt.get("type") in ("json_object", "json_schema"),
        "min_context": None,
    }


def _openai_error(message, type_="invalid_request_error", code=None):
    return {"error": {"message": message, "type": type_, "code": code}}


def _anthropic_error(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


class Handler(BaseHTTPRequestHandler):
    router = None       # Router, set in main
    catalog = None      # catalog document, set in main
    manifest_index = 0  # POSTs served so far, in manifest order
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_sse(self, text):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "keep-alive")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if is_record(value) else None

    def _free_models(self):
        """Every free model id in the catalog, once each, sorted."""
        ids = set()
        for entry in (self.catalog or {}).get("providers", []):
            for m in entry.get("models", []):
                if is_record(m) and m.get("free") is True and isinstance(m.get("id"), str):
                    ids.add(m["id"])
        return sorted(ids)

    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            self._send_json(404, _openai_error("no route GET %s" % self.path, "not_found", None))
            return
        ids = self._free_models()
        if isinstance(self.headers.get("x-api-key"), str):
            self._send_json(200, {
                "data": [{"type": "model", "id": mid, "display_name": mid,
                          "created_at": FIXED_ANTHROPIC_CREATED_AT, "free": True} for mid in ids],
                "has_more": False,
            })
        else:
            self._send_json(200, {
                "object": "list",
                "data": [{"id": mid, "object": "model", "created": FIXED_OPENAI_CREATED,
                          "owned_by": "aggregator", "free": True} for mid in ids],
            })

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path == "/v1/messages":
            self._handle_messages()
        else:
            wire = "anthropic" if path == "/v1/messages" else "openai"
            if wire == "anthropic":
                self._send_json(404, _anthropic_error("not_found_error", "no route POST %s" % path))
            else:
                self._send_json(404, _openai_error("no route POST %s" % path, "not_found", None))

    def _serve_one(self, body, client_api):
        """Route one client request through the router and render the answer."""
        with Handler.lock:
            manifest = self.router.brief.manifest
            idx = Handler.manifest_index
            Handler.manifest_index += 1
            if idx < len(manifest):
                entry = manifest[idx]
                requires = entry["requires"]
                request_id = entry["id"]
            else:
                requires = _requirements_from_body(body)
                request_id = None
            fmt = body.get("response_format") if is_record(body.get("response_format")) else {}
            json_mode = requires.get("json") is True or fmt.get("type") in ("json_object", "json_schema")
            stream = body.get("stream") is True
            ok, model, text, inp, outp = self.router.serve(body, client_api, requires, json_mode, request_id)
        if not ok:
            return False, None, None, None, None
        if client_api == "openai":
            if stream:
                return True, "sse", openai_sse_response(model, text, inp, outp), None, None
            return True, "json", openai_json_response(model, text, inp, outp), None, None
        if stream:
            return True, "sse", anthropic_sse_response(model, text, inp, outp), None, None
        return True, "json", anthropic_json_response(model, text, inp, outp), None, None

    def _handle_chat(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, _openai_error("the request body is not one JSON object"))
            return
        ok, kind, payload, _a, _b = self._serve_one(body, "openai")
        if not ok:
            self._send_json(503, _openai_error("no free model meets this request", "no_free_model", None))
            return
        if kind == "sse":
            self._send_sse(payload)
        else:
            self._send_json(200, payload)

    def _handle_messages(self):
        body = self._read_json()
        if body is None:
            self._send_json(400, _anthropic_error("invalid_request_error", "the request body is not one JSON object"))
            return
        ok, kind, payload, _a, _b = self._serve_one(body, "anthropic")
        if not ok:
            self._send_json(503, _anthropic_error("overloaded_error", "no free model meets this request"))
            return
        if kind == "sse":
            self._send_sse(payload)
        else:
            self._send_json(200, payload)


def register(env_url, port, catalog):
    status, body, raw = http_json("POST", env_url + "/register",
                                  {"content-type": "application/json"},
                                  {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                                  timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300:
        raise RuntimeError("POST %s/register answered HTTP %s: %s" % (env_url, status, (raw or "")[:500]))
    if is_record(body) and isinstance(body.get("requests"), int):
        log("registered: the workload holds %s requests" % body["requests"])


def poll_work(env_url, server):
    """Poll GET /work; shut the server down the moment it says done."""
    while True:
        try:
            status, body, _raw = http_json("GET", env_url + "/work", {}, timeout=2)
            if status == 200 and is_record(body) and body.get("done") is True:
                log("work done; exiting 0")
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_S)


def load_catalog(env_url, brief):
    probes = []
    for provider in brief.providers:
        ok, models = probe_provider(env_url, provider)
        if not ok:
            log("probe %s failed; it gets no traffic" % provider["id"])
        probes.append((ok, models))
    return build_catalog(brief, probes)


def main():
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    brief = Brief(prompt_file)
    if brief.error is not None:
        log(brief.error)
    if "--list" in sys.argv:
        if brief.error is not None or not env_url:
            catalog = {"providers": []}
        else:
            catalog = load_catalog(env_url, brief)
        text = json.dumps(catalog) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        return 0
    if not env_url:
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    catalog = load_catalog(env_url, brief)
    router = Router(brief, catalog)
    log("plan: %s" % json.dumps(router.plan))
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    Handler.router = router
    Handler.catalog = catalog
    Handler.manifest_index = 0
    address = server.server_address
    port = address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%s with %s providers and %s free models"
        % (port, len(catalog["providers"]),
           len({m["id"] for p in catalog["providers"] for m in p["models"] if m["free"]})))
    register(env_url, port, catalog)
    poller = threading.Thread(target=poll_work, args=(env_url, server), daemon=True)
    poller.start()
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
