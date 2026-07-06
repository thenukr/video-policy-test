from __future__ import annotations

import html
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def build_gallery(results_dir: Path) -> Path:
    videos = sorted(results_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    cards = []
    for video in videos:
        relative = video.relative_to(results_dir)
        label = html.escape(str(relative))
        cards.append(
            f'<article><video controls preload="metadata" src="{html.escape(str(relative))}"></video>'
            f"<p>{label}</p></article>"
        )
    content = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Video policy rollouts</title>
<style>
body {{ background:#111; color:#eee; font:15px system-ui; margin:2rem }}
main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:1.2rem }}
article {{ background:#1d1d1d; padding:.8rem; border-radius:10px }}
video {{ width:100%; border-radius:6px }} p {{ overflow-wrap:anywhere }}
</style></head><body><h1>Video policy rollouts</h1>
<p>{len(videos)} video(s) under {html.escape(str(results_dir))}</p><main>{''.join(cards)}</main></body></html>
"""
    index = results_dir / "index.html"
    index.write_text(content, encoding="utf-8")
    return index


def serve_gallery(results_dir: Path, host: str, port: int) -> None:
    results_dir = results_dir.resolve()
    index = build_gallery(results_dir)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(results_dir), **kwargs)

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                build_gallery(results_dir)
            super().do_GET()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving {index} at http://{host}:{port}/ (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def read_summary(results_dir: Path) -> dict:
    path = results_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
