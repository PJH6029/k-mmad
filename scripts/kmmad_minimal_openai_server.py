#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for K-MMAD translation smoke.

This is intentionally small and dependency-light: it uses Python's stdlib HTTP
server plus Hugging Face Transformers so an MLXP reservation can serve an
open-source instruction model even when vLLM is not preinstalled in the image.
It implements only the endpoints needed by `kmmad_translate_smoke.py`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class ChatModel:
    def __init__(
        self,
        model_name: str,
        max_model_length: int | None = None,
        *,
        backend: str = "auto",
        source_lang_code: str = "en",
        target_lang_code: str = "ko",
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")

        self.model_name = model_name
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.backend = normalize_backend(backend, model_name)
        self.source_lang_code = source_lang_code
        self.target_lang_code = target_lang_code
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            load_kwargs["revision"] = revision
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"

        if self.backend == "translategemma":
            auto_processor = getattr(transformers, "AutoProcessor")
            auto_model_for_image_text_to_text = getattr(
                transformers,
                "AutoModelForImageTextToText",
                getattr(transformers, "AutoModelForMultimodalLM", None),
            )
            if auto_model_for_image_text_to_text is None:
                raise RuntimeError(
                    "TranslateGemma backend requires transformers AutoModelForImageTextToText "
                    "or AutoModelForMultimodalLM."
                )
            self.processor = auto_processor.from_pretrained(model_name, **load_kwargs)
            kwargs = {"device_map": "auto", "low_cpu_mem_usage": True, **load_kwargs}
            if dtype != "auto":
                kwargs["torch_dtype"] = dtype
            self.model = auto_model_for_image_text_to_text.from_pretrained(model_name, **kwargs)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            self.stop_token_ids = []
            self.model.eval()
            return

        auto_tokenizer = getattr(transformers, "AutoTokenizer")
        auto_model_for_causal_lm = getattr(transformers, "AutoModelForCausalLM")
        self.tokenizer = auto_tokenizer.from_pretrained(model_name, **load_kwargs)
        causal_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            **load_kwargs,
        }
        if max_model_length:
            causal_kwargs["max_position_embeddings"] = max_model_length
        self.model = auto_model_for_causal_lm.from_pretrained(model_name, **causal_kwargs)
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()
        stop_ids = [self.tokenizer.eos_token_id]
        for token in ("<|eot_id|>", "<|im_end|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in stop_ids:
                stop_ids.append(token_id)
        self.stop_token_ids = [token_id for token_id in stop_ids if token_id is not None]

    def chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        if self.backend == "translategemma":
            return self.translate_gemma_chat(messages, max_tokens=max_tokens)
        return self.causal_lm_chat(messages, max_tokens=max_tokens, temperature=temperature)

    def causal_lm_chat(self, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        torch = importlib.import_module("torch")

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
        do_sample = temperature > 0
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                eos_token_id=self.stop_token_ids or self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def translate_gemma_chat(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        """Translate text through TranslateGemma's required single-entry user template.

        The OpenAI-compatible smoke client sends a system prompt plus a plain
        user string. TranslateGemma officially supports only user/assistant
        roles and expects the user content list to contain exactly one typed
        translation item, so the server adapts the final user text into that
        shape instead of forwarding the generic chat prompt verbatim.
        """

        torch = importlib.import_module("torch")
        text = final_user_text(messages)
        translate_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": self.source_lang_code,
                        "target_lang_code": self.target_lang_code,
                        "text": text,
                    }
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            translate_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device, dtype=dtype) if dtype is not None else inputs.to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens)
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()


def normalize_backend(backend: str, model_name: str) -> str:
    value = backend.strip().lower().replace("-", "_")
    if value == "auto":
        return "translategemma" if "translategemma" in model_name.lower() else "causal_lm"
    aliases = {
        "causal": "causal_lm",
        "causal_lm": "causal_lm",
        "text_generation": "causal_lm",
        "translate_gemma": "translategemma",
        "translategemma": "translategemma",
    }
    if value not in aliases:
        raise ValueError(f"unsupported backend: {backend}")
    return aliases[value]


def final_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                return "\n".join(text_parts)
    return ""


def make_handler(chat_model: ChatModel) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "KMMADMinimalOpenAI/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.log_date_time_string()} {self.address_string()} {format % args}", flush=True)

        def write_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path.rstrip("/") in {"", "/health", "/v1/models"}:
                if self.path.rstrip("/") == "/v1/models":
                    self.write_json(200, {"object": "list", "data": [{"id": chat_model.model_name, "object": "model"}]})
                else:
                    self.write_json(
                        200,
                        {
                            "status": "ok",
                            "model": chat_model.model_name,
                            "backend": chat_model.backend,
                            "revision": chat_model.revision,
                            "trust_remote_code": chat_model.trust_remote_code,
                        },
                    )
                return
            self.write_json(404, {"error": {"message": f"unknown endpoint: {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path.rstrip("/") != "/v1/chat/completions":
                self.write_json(404, {"error": {"message": f"unknown endpoint: {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload["messages"]
                max_tokens = int(payload.get("max_tokens", 512))
                temperature = float(payload.get("temperature", 0.0))
                content = chat_model.chat(messages, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # pragma: no cover - exercised on cluster failures
                traceback.print_exc()
                self.write_json(500, {"error": {"message": str(exc), "type": type(exc).__name__}})
                return
            now = int(time.time())
            self.write_json(
                200,
                {
                    "id": f"chatcmpl-kmmad-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": chat_model.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "causal-lm", "translategemma"),
        help="Model adapter. auto selects TranslateGemma for google/translategemma-* models.",
    )
    parser.add_argument("--source-lang-code", default="en", help="TranslateGemma source language code")
    parser.add_argument("--target-lang-code", default="ko", help="TranslateGemma target language code")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-length", type=int, default=None)
    parser.add_argument("--revision", default=None, help="Optional pinned model revision/commit")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow execution of model repository code. Keep disabled unless a pinned, reviewed model requires it.",
    )
    args = parser.parse_args(argv)

    chat_model = ChatModel(
        args.model,
        max_model_length=args.max_model_length,
        backend=args.backend,
        source_lang_code=args.source_lang_code,
        target_lang_code=args.target_lang_code,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(chat_model))
    print(f"serving {args.model} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
