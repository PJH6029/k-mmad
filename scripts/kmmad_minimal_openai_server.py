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
        trust_remote_code: bool = False,
        revision: str | None = None,
    ) -> None:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        auto_tokenizer = getattr(transformers, "AutoTokenizer")
        auto_model_for_causal_lm = getattr(transformers, "AutoModelForCausalLM")

        self.model_name = model_name
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if revision:
            load_kwargs["revision"] = revision
        self.tokenizer = auto_tokenizer.from_pretrained(model_name, **load_kwargs)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            **load_kwargs,
        }
        if max_model_length:
            kwargs["max_position_embeddings"] = max_model_length
        self.model = auto_model_for_causal_lm.from_pretrained(model_name, **kwargs)
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
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(chat_model))
    print(f"serving {args.model} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
