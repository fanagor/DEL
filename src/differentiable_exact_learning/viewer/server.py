"""Small loopback-only viewer server using the Python standard library."""

import argparse
import json
import secrets
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files

from .controller import Session, load_controller


def make_server(source, text=None, *, width=21, height=9, max_steps=100000, port=0):
    """Return an unstarted server, useful for embedding and deterministic tests."""
    for value, name in [(width, "width"), (height, "height")]:
        if not isinstance(value, int) or isinstance(value, bool) or not 3 <= value <= 101:
            raise ValueError(f"{name} must contain 3–101 cells")
    if text is None:
        text = (
            "101101"
            if source.expert.environment == "teleporting" or source.binary_tape
            else ("101+11" if source.expert.task == "addition" else "11x101")
        )
    Session(source, text, max_steps=max_steps)  # Fail before opening the browser.
    token, sessions, lock = secrets.token_urlsafe(24), OrderedDict(), threading.Lock()
    assets = files(__package__).joinpath("static")
    bootstrap = {
        "label": source.label,
        "environment": source.expert.environment,
        "model": source.model is not None,
        "tasks": source.tasks(),
        "task": source.task,
        "input": text,
        "width": width,
        "height": height,
        "max_steps": max_steps,
        "token": token,
    }
    payload = json.dumps(bootstrap).replace("<", "\\u003c")
    page = assets.joinpath("index.html").read_text().replace("__BOOTSTRAP__", payload).encode()
    static = {"/": (page, "text/html; charset=utf-8")}
    for name, mime in [
        ("style.css", "text/css"),
        ("viewer.mjs", "text/javascript"),
        ("camera.mjs", "text/javascript"),
    ]:
        static["/" + name] = (assets.joinpath(name).read_bytes(), mime + "; charset=utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body, mime="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; "
                "style-src 'self'; connect-src 'self'; img-src 'self' data:; "
                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A closed/reloaded tab does not invalidate the server.

        def trusted_host(self):
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def do_GET(self):
            if not self.trusted_host():
                return self.reply(403, {"error": "Local viewer origin required"})
            if self.path not in static:
                return self.reply(404, {"error": "Not found"})
            body, mime = static[self.path]
            self.reply(200, body, mime)

        def do_POST(self):
            if not self.trusted_host() or not secrets.compare_digest(
                self.headers.get("X-Viewer-Token", ""), token
            ):
                return self.reply(403, {"error": "Reload the local viewer to reconnect"})
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers['Host']}":
                return self.reply(403, {"error": "Local viewer origin required"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Request body must contain 1–32768 bytes")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                with lock:
                    if self.path == "/api/reset":
                        session = Session(
                            source, request["input"], task=request.get("task"), max_steps=max_steps
                        )
                        previous = request.get("session")
                        if previous in sessions:
                            del sessions[previous]
                        key = secrets.token_urlsafe(18)
                        sessions[key] = session
                        while len(sessions) > 8:
                            sessions.popitem(last=False)
                        result = {"session": key, **session.snapshot()}
                    elif self.path == "/api/step":
                        key = request.get("session")
                        if key not in sessions:
                            raise ValueError("Session expired. Reset to start again.")
                        count = request.get("count", 1)
                        if (
                            not isinstance(count, int)
                            or isinstance(count, bool)
                            or not 1 <= count <= 64
                        ):
                            raise ValueError("Each request must advance 1–64 steps")
                        session = sessions[key]
                        sessions.move_to_end(key)
                        events = []
                        for _ in range(count):
                            event = session.advance()
                            if event is None:
                                break
                            events.append(event)
                        result = {
                            "events": events,
                            "status": session.status,
                            "message": session.message,
                        }
                    else:
                        return self.reply(404, {"error": "Not found"})
                self.reply(200, result)
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def open_viewer(
    controller,
    text=None,
    *,
    config=None,
    task=None,
    device="cpu",
    width=21,
    height=9,
    max_steps=100000,
    port=0,
    open_browser=True,
):
    """Serve an interactive viewer until Ctrl-C. Accept selectors, objects, or weights."""
    source = load_controller(controller, config=config, task=task, device=device)
    server = make_server(source, text, width=width, height=height, max_steps=max_steps, port=port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Viewer: {url}\nController: {source.label}\nPress Ctrl-C to close.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def add_arguments(parser):
    parser.add_argument(
        "controller", help="grid:add2, tm:mult2, tm:parity, tp:copy, recipe, run, or weights"
    )
    parser.add_argument("input", nargs="?", help="Initial arithmetic expression or binary string")
    parser.add_argument("--config", help="Experiment YAML for raw state_dict weights")
    parser.add_argument("--task", help="Binary program for a teleporting controller")
    parser.add_argument("--width", type=int, default=21, help="Visible columns (3–101)")
    parser.add_argument("--height", type=int, default=9, help="Visible rows (3–101)")
    parser.add_argument("--max-steps", type=int, default=100000, help="Execution bound per input")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 chooses an unused port")
    parser.add_argument(
        "--no-browser", action="store_true", help="Print the URL without opening a browser"
    )


def run_arguments(args):
    open_viewer(
        args.controller,
        args.input,
        config=args.config,
        task=args.task,
        device=args.device,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        port=args.port,
        open_browser=not args.no_browser,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run_arguments(parser.parse_args(argv))
