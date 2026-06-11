from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kmmad_common import find_sensitive_strings, sanitize_jsonable, sanitize_text  # noqa: E402
from kmmad_run_record import validate as validate_run_record  # noqa: E402
from kmmad_translate_smoke import (  # noqa: E402
    build_request_headers,
    call_openai_compatible,
    sanitized_headers,
)


class CapturingHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "content_type": self.headers.get("Content-Type"),
                "body": body,
            }
        )
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "test-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "한국어 계약 테스트 응답"},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def base_config(url: str, *, auth_mode: str = "none", env_name: str = "KMMAD_TEST_OPENAI_API_KEY") -> dict[str, Any]:
    return {
        "inference": {
            "endpoint_provider": "contract_test",
            "auth_mode": auth_mode,
            "api_key_env": env_name,
            "openai_compatible_base_url": url,
            "model": "contract-test-model",
            "temperature": 0.0,
            "max_tokens": 32,
            "timeout_seconds": 5,
        }
    }


class EndpointAuthContractTest(unittest.TestCase):
    def setUp(self) -> None:
        CapturingHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), CapturingHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        os.environ.pop("KMMAD_TEST_OPENAI_API_KEY", None)

    def test_bearer_env_contract_sends_authorization_and_parses_response(self) -> None:
        secret = "test-secret-token-123"
        os.environ["KMMAD_TEST_OPENAI_API_KEY"] = secret
        config = base_config(self.base_url, auth_mode="bearer_env")

        result = call_openai_compatible(config, "Translate this.")

        self.assertEqual(result, "한국어 계약 테스트 응답")
        self.assertEqual(len(CapturingHandler.requests), 1)
        captured = CapturingHandler.requests[0]
        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertEqual(captured["body"]["model"], "contract-test-model")
        self.assertIn("messages", captured["body"])
        self.assertIn("max_tokens", captured["body"])

        headers = build_request_headers(config)
        redacted = sanitized_headers(headers)
        self.assertEqual(redacted["Authorization"], "Bearer <redacted>")
        persisted = sanitize_jsonable({"command": f"curl -H 'Authorization: Bearer {secret}'"}, [secret])
        self.assertEqual(find_sensitive_strings(persisted, extra_secrets=[secret]), [])
        self.assertNotIn(secret, json.dumps(persisted))

    def test_none_auth_mode_omits_authorization(self) -> None:
        config = base_config(self.base_url, auth_mode="none")
        result = call_openai_compatible(config, "Translate this.")
        self.assertEqual(result, "한국어 계약 테스트 응답")
        self.assertIsNone(CapturingHandler.requests[0]["authorization"])

    def test_missing_bearer_env_fails_without_secret_value(self) -> None:
        config = base_config(self.base_url, auth_mode="bearer_env")
        with self.assertRaisesRegex(RuntimeError, "KMMAD_TEST_OPENAI_API_KEY") as ctx:
            build_request_headers(config)
        self.assertNotIn("Bearer", str(ctx.exception))
        self.assertNotIn("test-secret-token", str(ctx.exception))

    def test_run_record_validation_rejects_unredacted_secret_without_echoing_it(self) -> None:
        secret = "sk-testsecret1234567890"
        record = {
            "run_id": "contract-test",
            "git_sha": "HEAD",
            "commands": [f"curl -H 'Authorization: Bearer {secret}' http://127.0.0.1"],
            "dataset": {},
            "artifacts": {},
            "reservation": {},
            "image_runtime": {},
            "model": {},
            "status": "failed",
        }

        errors = validate_run_record(record)

        joined = "\n".join(errors)
        self.assertIn("$.commands[0]", joined)
        self.assertNotIn(secret, joined)

    def test_mlxp_json_token_fields_are_detected_and_redacted(self) -> None:
        reservation = {
            "reservation_id": "rsv-test",
            "status": "running",
            "jupyter_token": "mlxp-jupyter-secret",
            "nested": {
                "access_token": "oauth-access-secret",
                "token": "generic-secret",
                "custom_token": "custom-secret",
                "client_secret": "client-secret",
                "password": "password-secret",
            },
            "auth_secret_present": True,
            "secret_findings": [],
        }

        findings = find_sensitive_strings(reservation)
        sanitized = sanitize_jsonable(reservation)

        self.assertEqual(
            findings,
            [
                "$.jupyter_token",
                "$.nested.access_token",
                "$.nested.token",
                "$.nested.custom_token",
                "$.nested.client_secret",
                "$.nested.password",
            ],
        )
        self.assertEqual(sanitized["jupyter_token"], "<redacted>")
        self.assertEqual(sanitized["nested"]["access_token"], "<redacted>")
        self.assertEqual(sanitized["nested"]["token"], "<redacted>")
        self.assertEqual(sanitized["nested"]["custom_token"], "<redacted>")
        self.assertEqual(sanitized["nested"]["client_secret"], "<redacted>")
        self.assertEqual(sanitized["nested"]["password"], "<redacted>")
        self.assertIs(sanitized["auth_secret_present"], True)
        self.assertEqual(sanitized["secret_findings"], [])
        self.assertEqual(find_sensitive_strings(sanitized), [])
        self.assertNotIn("mlxp-jupyter-secret", json.dumps(sanitized))

    def test_quoted_token_text_is_sanitized(self) -> None:
        text = '{"jupyter_token":"mlxp-jupyter-secret","access_token": "oauth-access-secret"}'
        sanitized = sanitize_text(text)
        self.assertNotIn("mlxp-jupyter-secret", sanitized)
        self.assertNotIn("oauth-access-secret", sanitized)
        self.assertIn('"jupyter_token":"<redacted>"', sanitized)


if __name__ == "__main__":
    unittest.main()
