#!/usr/bin/env python3
"""Serve the dashboard directory (index.html + data.json) on 127.0.0.1, tolerant of a noisy port-forward.

  dashboard_server.py --dir /data/teutonic/www --port 8787

`python -m http.server` rejected requests that arrived through the user's tunnel with framing bytes glued in front of
the request line ("\\x09" + 12 NUL bytes before "GET /data.json ...", 2026-10-09: 400 "Bad request syntax" / 501
"Unsupported method"), so page loads and the page's 10 s data polls failed at random. This handler drops any leading
tab/NUL/CR/LF bytes before parsing, never caches, and answers HTTP/1.1 so browsers and proxies render the page.
"""
from __future__ import annotations

import argparse
import functools
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

JUNK = re.compile(rb"^[\x00\t\r\n ]+")


class Handler(SimpleHTTPRequestHandler):
    # HTTP/1.1: browsers and the RunPod / Cloudflare proxies drop or blank an HTTP/1.0 response.
    # Junk-byte stripping below still covers the noisy SSH tunnel that motivated the custom server.
    protocol_version = "HTTP/1.1"
    # explicit UTF-8: without it browsers decode the page as Windows-1252 and the ✅/❌ marks show as "âœ…"/"âŒ"
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8"}

    def parse_request(self) -> bool:
        line = JUNK.sub(b"", self.raw_requestline)
        if line != self.raw_requestline:
            self.raw_requestline = line
            self.requestline = ""  # re-derived by the base class from raw_requestline
        return super().parse_request()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # only errors: one line per 10 s poll is noise
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data/teutonic/www")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), functools.partial(Handler, directory=args.dir))
    print(f"serving {args.dir} on http://{args.bind}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
