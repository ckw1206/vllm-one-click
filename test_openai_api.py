#!/usr/bin/env python3
"""
Quick test for vLLM OpenAI-compatible API.

Use SSH tunnel when the server is not on your network:
  ./tunnel_vllm.sh
  python test_openai_api.py   # uses localhost:8000

Setup (use a venv to avoid system Python restrictions):
  python3 -m venv .venv
  source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
  pip install -r requirements.txt

Run:
  python test_openai_api.py [--host localhost] [--port 8000]
  python test_openai_api.py --host 10.10.1.153   # direct if on same network

Curl (with tunnel; key required): curl -s -H "Authorization: Bearer not-needed" http://127.0.0.1:8000/v1/models
"""
import argparse
import os
import sys

try:
    from openai import OpenAI
    from openai import APIConnectionError
    from openai import AuthenticationError
except ImportError:
    print("Install the OpenAI client. Using a venv is recommended:")
    print("  python3 -m venv .venv && source .venv/bin/activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="Test vLLM OpenAI-compatible API")
    p.add_argument("--host", default="127.0.0.1", help="vLLM server (default: 127.0.0.1 for tunnel)")
    p.add_argument("--port", type=int, default=8000, help="vLLM server port")
    p.add_argument("--model", default=None, help="Model name (default: use server default)")
    args = p.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    print(f"Connecting to {base_url} ...")

    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=120.0,  # vLLM first request can be slow
    )

    # Preflight: check if server responds (avoids vague "disconnected" on tunnel/server issues)
    try:
        client.models.list()
    except APIConnectionError as e:
        cause = getattr(e, "__cause__", None)
        print(f"Connection failed (preflight): {e}")
        if cause:
            print(f"Cause: {cause}")
        print("Check: 1) Tunnel running? ./tunnel_vllm.sh  2) On server: curl -s http://127.0.0.1:8000/v1/models")
        sys.exit(1)
    except AuthenticationError:
        pass  # 401 on /models is ok; try chat anyway

    # Chat completion
    kwargs = {"model": args.model} if args.model else {}
    try:
        resp = client.chat.completions.create(
            model=kwargs.get("model") or "MiniMax-M2.1",
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=64,
        )
    except APIConnectionError as e:
        print(f"Connection failed: {e}")
        if getattr(e, "__cause__", None):
            print(f"Cause: {e.__cause__}")
        print("If cause is 'Server disconnected without sending a response':")
        print("  On server run: curl -s http://127.0.0.1:8000/v1/models")
        print("  If that fails, vLLM may still be loading or check docker logs vllm-openai")
        if args.host not in ("127.0.0.1", "localhost"):
            print("Tip: use SSH tunnel then connect to 127.0.0.1:")
            print("  ./tunnel_vllm.sh   # in another terminal")
            print("  python test_openai_api.py")
        sys.exit(1)
    except AuthenticationError as e:
        print(f"Auth failed: {e}")
        print("Tip: start the server with install.sh (it passes --api-key 'not-needed').")
        print("  Or set OPENAI_API_KEY to the key the server expects.")
        sys.exit(1)

    content = resp.choices[0].message.content
    print("Response:", content)
    print("Done.")


if __name__ == "__main__":
    main()
