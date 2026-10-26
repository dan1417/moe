#!/usr/bin/env python3
"""Poll the Teutonic dashboard; log king changes and challengers entering the queue / evaluation.

Writes one line per change to stdout (run under pm2 as `kingwatch`) and, on a new king, the file
/workspace/teutonic/KING_CHANGED (json of the new king) so training loops and humans notice.
  python scripts/data/king_watch.py [--poll 120]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

URL = "https://teutonic.ai/dashboard.json"
FLAG = Path("/workspace/teutonic/KING_CHANGED")


def fetch() -> dict:
    request = urllib.request.Request(URL, headers={"User-Agent": "teutonic-kingwatch/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll", type=int, default=120)
    args = parser.parse_args()
    last_king = last_eval = last_queue = None
    while True:
        try:
            board = fetch()
        except Exception as exc:  # noqa: BLE001 - network hiccups must not kill the watcher
            print(time.strftime("%H:%M:%S"), "fetch failed:", exc, flush=True)
            time.sleep(args.poll)
            continue
        king = board.get("king") or {}
        king_id = (king.get("reign_number"), king.get("king_digest"))
        if king_id != last_king:
            print(time.strftime("%H:%M:%S"), "KING", json.dumps({k: king.get(k) for k in ("reign_number", "model_repo", "king_digest", "coldkey", "crowned_at", "mu_hat", "lcb")}), flush=True)
            if last_king is not None:
                FLAG.write_text(json.dumps(king, indent=1))
            last_king = king_id
        current = board.get("current_eval")
        eval_id = json.dumps(current, sort_keys=True)[:300] if current else None
        if eval_id != last_eval:
            print(time.strftime("%H:%M:%S"), "CURRENT_EVAL", eval_id, flush=True)
            last_eval = eval_id
        queue = board.get("queue") or []
        queue_id = json.dumps(queue, sort_keys=True)[:500]
        if queue_id != last_queue:
            print(time.strftime("%H:%M:%S"), "QUEUE", len(queue), queue_id, flush=True)
            last_queue = queue_id
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
