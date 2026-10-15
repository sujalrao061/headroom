"""Loopback-only JSON API for the fixed guest demo."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import run
from .nebius import propose, propose_attention
from .workloads import CASES


class Handler(BaseHTTPRequestHandler):
    _job_slot = threading.BoundedSemaphore(1)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"ok": True, "nebius_configured": bool(os.getenv("NEBIUS_API_KEY")), "supported_cases": list(CASES), "attention_supported": True})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path not in ("/run", "/attention"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json({"error": "Invalid request length"}, 400)
            return
        if length < 0 or length > 4096:
            self.send_error(413)
            return
        if not self._job_slot.acquire(blocking=False):
            self.send_json({"error": "GPU runner is busy. Try again after the current run finishes."}, 429)
            return
        try:
            body = json.loads(self.rfile.read(length))
            if self.path == "/attention":
                from scripts.benchmark_attention import capture
                size = int(body.get("size", 1536))
                width = int(body.get("width", 64))
                memory = int(body.get("memory_budget_mib", 8))
                if not 64 <= size <= 1536 or not 8 <= width <= 128 or not 1 <= memory <= 4096:
                    raise ValueError("Inputs outside allowed attention demo range")
                result = capture(size, width, memory)
                result["nemotron"] = propose_attention(result)
                result["model_agrees"] = result["nemotron"]["strategy"] == result["recommended"] if result["nemotron"]["status"] == "live" else None
                self.send_json(result)
                return
            result = run(body.get("case_id", "nearest"), int(body.get("size", 1536)), int(body.get("memory_limit_mb", 8)), int(body.get("latency_limit_ms", 1000)))
            result["nemotron"] = propose(result["case_id"], result["source"], result["contract"], result["size"], {
                "baseline": result["baseline"], "candidate": result["candidate"], "torch_cdist": result["library"], "checks": result["checks"],
                "memory_limit_mb": result["memory_limit_mb"], "latency_limit_ms": result["latency_limit_ms"],
                "failure_kind": result["failure_kind"],
            })
            expected_strategy = CASES[result["case_id"]]["strategy"]
            if result["nemotron"]["status"] == "live" and result["nemotron"]["strategy"] != expected_strategy:
                result["accepted"] = False
                result["reasons"].insert(0, "Nemotron did not propose the contract-preserving bounded strategy; the measured candidate remains a counterfactual test, not an accepted model repair.")
            result["candidate_origin"] = "Nemotron strategy with independent validation" if result["accepted"] and result["nemotron"]["status"] == "live" and result["nemotron"]["strategy"] == expected_strategy else "fixed local candidate"
            self.send_json(result)
        except (ValueError, TypeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"Runner failed: {type(exc).__name__}"}, 500)
        finally:
            self._job_slot.release()

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
