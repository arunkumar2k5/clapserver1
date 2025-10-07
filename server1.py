import asyncio, json, os, traceback
import websockets
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv

# OpenAI SDK (new-style)
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BIND_HOST = os.getenv("BIND_HOST", "0.0.0.0")
BIND_PORT = int(os.getenv("BIND_PORT", "8765"))

if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY in environment or .env")

oai = OpenAI(api_key=OPENAI_API_KEY)

SERVER_NAME = "sample-mcp-server"
CAPS = ["llm.generate"]

async def handle(ws):
    # 1) wait for initialize
    init_msg = await ws.recv()
    try:
        init = json.loads(init_msg)
    except json.JSONDecodeError:
        await ws.send(json.dumps({
            "type": "result",
            "id": "init",
            "ok": False,
            "error": {"code": "BadJSON", "message": "Initialize must be JSON"}
        }))
        return

    # optional: validate initialize
    await ws.send(json.dumps({
        "type": "ready",
        "server": SERVER_NAME,
        "capabilities": CAPS
    }))

    # 2) loop for requests
    while True:
        try:
            raw = await ws.recv()
        except ConnectionClosed:
            break

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send(json.dumps({
                "type": "result",
                "id": "unknown",
                "ok": False,
                "error": {"code": "BadJSON", "message": "Request must be JSON"}
            }))
            continue

        if msg.get("type") != "request":
            await ws.send(json.dumps({
                "type": "result",
                "id": msg.get("id", "unknown"),
                "ok": False,
                "error": {"code": "BadRequest", "message": "Expected type=request"}
            }))
            continue

        req_id = msg.get("id", "unknown")
        method = msg.get("method")
        params = msg.get("params", {})

        if method == "llm.generate":
            # Extract params with sane defaults
            prompt = params.get("prompt", "")
            system = params.get("system", "You are a helpful assistant.")
            model = params.get("model", OPENAI_MODEL)
            temperature = float(params.get("temperature", 0.2))
            format_ = params.get("format", "markdown")

            if not prompt.strip():
                await ws.send(json.dumps({
                    "type": "result",
                    "id": req_id,
                    "ok": False,
                    "error": {"code": "BadRequest", "message": "prompt is required"}
                }))
                continue

            try:
                # Call OpenAI chat completions (simple, non-streaming)
                chat = oai.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ]
                )
                text = chat.choices[0].message.content
                usage = getattr(chat, "usage", None)
                await ws.send(json.dumps({
                    "type": "result",
                    "id": req_id,
                    "ok": True,
                    "data": {
                        "text": text,
                        "format": format_,
                        "usage": usage.model_dump() if usage else {}
                    }
                }))
            except Exception as e:
                await ws.send(json.dumps({
                    "type": "result",
                    "id": req_id,
                    "ok": False,
                    "error": {
                        "code": "LLMError",
                        "message": str(e),
                        "trace": traceback.format_exc().splitlines()[-5:]
                    }
                }))

        else:
            await ws.send(json.dumps({
                "type": "result",
                "id": req_id,
                "ok": False,
                "error": {"code": "NoSuchMethod", "message": f"Unknown method {method}"}
            }))

async def main():
    print(f"[{SERVER_NAME}] listening on ws://{BIND_HOST}:{BIND_PORT}")
    async with websockets.serve(handle, BIND_HOST, BIND_PORT, max_size=2**23):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
