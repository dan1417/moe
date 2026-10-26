#!/usr/bin/env python3
"""Live Teutonic dashboard poller and new-king alerter."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_king import DASHBOARD_URL, DEFAULT_TIMEOUT, download_king, get_king_digest

log = logging.getLogger(__name__)

DEFAULT_MANIFEST_URL = "https://teutonic.ai/datasets/manifest.json"
DEFAULT_POLL_INTERVAL_SEC = 60.0
DEFAULT_STATE_PATH = Path(".teutonic_monitor_state.json")
USER_AGENT = "teutonic-miner/1.0"


def _fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.json()


@dataclass
class MonitorState:
    """Persisted monitor state across restarts."""

    last_king_digest: str | None = None
    last_config_version: str | None = None
    last_poll_at: str | None = None

    def __repr__(self) -> str:
        return (
            f"MonitorState(king_digest={self.last_king_digest!r}, "
            f"config_version={self.last_config_version!r}, last_poll_at={self.last_poll_at!r})"
        )


@dataclass
class PollResult:
    """Outcome of a single dashboard/manifest poll."""

    king_digest: str
    king_changed: bool
    config_version: str
    manifest_changed: bool
    king_meta: dict[str, Any]
    dashboard_updated_at: str | None

    def __repr__(self) -> str:
        return (
            f"PollResult(king_digest={self.king_digest!r}, king_changed={self.king_changed}, "
            f"config_version={self.config_version!r}, manifest_changed={self.manifest_changed})"
        )


class DashboardMonitor:
    """Poll Teutonic dashboard and dataset manifest for king or dataset changes."""

    def __init__(
        self,
        dashboard_url: str = DASHBOARD_URL,
        manifest_url: str = DEFAULT_MANIFEST_URL,
        state_path: Path = DEFAULT_STATE_PATH,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        timeout: float = DEFAULT_TIMEOUT,
        auto_download: bool = False,
        king_output_dir: Path = Path("./king"),
        beep_on_change: bool = False,
    ) -> None:
        self.dashboard_url = dashboard_url
        self.manifest_url = manifest_url
        self.state_path = state_path
        self.poll_interval_sec = poll_interval_sec
        self.timeout = timeout
        self.auto_download = auto_download
        self.king_output_dir = king_output_dir
        self.beep_on_change = beep_on_change

    def __repr__(self) -> str:
        return (
            f"DashboardMonitor(dashboard_url={self.dashboard_url!r}, "
            f"manifest_url={self.manifest_url!r}, poll_interval_sec={self.poll_interval_sec}, "
            f"auto_download={self.auto_download}, king_output_dir={self.king_output_dir!r})"
        )

    def load_state(self, timeout: float = DEFAULT_TIMEOUT) -> MonitorState:
        del timeout
        if not self.state_path.exists():
            return MonitorState()
        try:
            payload = json.loads(self.state_path.read_text())
            return MonitorState(
                last_king_digest=payload.get("last_king_digest"),
                last_config_version=payload.get("last_config_version"),
                last_poll_at=payload.get("last_poll_at"),
            )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            log.warning("Could not read monitor state at %s: %s — starting fresh", self.state_path, exc)
            return MonitorState()

    def save_state(self, state: MonitorState, timeout: float = DEFAULT_TIMEOUT) -> None:
        del timeout
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(state), indent=2))

    def fetch_manifest_config_version(self, timeout: float | None = None) -> str:
        request_timeout = self.timeout if timeout is None else timeout
        payload = _fetch_json(self.manifest_url, timeout=request_timeout)
        return str(payload.get("config_version", ""))

    def poll_once(self, state: MonitorState, timeout: float | None = None) -> tuple[PollResult, MonitorState]:
        request_timeout = self.timeout if timeout is None else timeout
        king_meta = get_king_digest(dashboard_url=self.dashboard_url, timeout=request_timeout)
        king_digest = str(king_meta["digest"])
        config_version = self.fetch_manifest_config_version(timeout=request_timeout)

        dashboard = _fetch_json(self.dashboard_url, timeout=request_timeout)
        dashboard_updated_at = dashboard.get("updated_at")

        king_changed = state.last_king_digest is not None and king_digest != state.last_king_digest
        manifest_changed = (
            state.last_config_version is not None
            and config_version != ""
            and config_version != state.last_config_version
        )

        new_state = MonitorState(
            last_king_digest=king_digest,
            last_config_version=config_version or state.last_config_version,
            last_poll_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        result = PollResult(
            king_digest=king_digest,
            king_changed=king_changed,
            config_version=config_version,
            manifest_changed=manifest_changed,
            king_meta=king_meta,
            dashboard_updated_at=str(dashboard_updated_at) if dashboard_updated_at is not None else None,
        )
        return result, new_state

    def _alert(self, message: str) -> None:
        log.warning(message)
        if self.beep_on_change:
            sys.stderr.write("\a")
            sys.stderr.flush()

    def handle_poll_result(self, result: PollResult, previous_state: MonitorState) -> bool:
        """Process poll result. Returns True if a king change was detected."""
        if previous_state.last_king_digest is None:
            log.info(
                "Monitor initialized: king=%s reign=%s loss=%.4f",
                result.king_digest[:12],
                result.king_meta.get("model_name", "unknown"),
                float(result.king_meta.get("avg_king_loss") or 0.0),
            )
            return False

        if result.manifest_changed:
            self._alert(
                "Dataset manifest config_version changed: "
                f"{previous_state.last_config_version} -> {result.config_version}. "
                "Review dataset proportions before the next training run."
            )

        if not result.king_changed:
            log.debug(
                "No king change (digest=%s, dashboard_updated_at=%s)",
                result.king_digest[:12],
                result.dashboard_updated_at,
            )
            return False

        self._alert(
            "NEW KING DETECTED: "
            f"{previous_state.last_king_digest[:12]}... -> {result.king_digest[:12]}... "
            f"model={result.king_meta.get('model_name')} "
            f"hotkey={result.king_meta.get('hotkey')} "
            f"avg_king_loss={result.king_meta.get('avg_king_loss')}"
        )

        if self.auto_download:
            log.info("Auto-downloading new king to %s", self.king_output_dir)
            download_king(
                output_dir=self.king_output_dir,
                dashboard_url=self.dashboard_url,
                timeout=self.timeout,
                verify_only=False,
            )
        return True

    def run(self, once: bool = False) -> int:
        """Poll dashboard until interrupted, or once when `once=True`."""
        state = self.load_state()
        log.info("Starting %s (state=%s)", self, state)

        while True:
            try:
                result, new_state = self.poll_once(state)
                king_changed = self.handle_poll_result(result, state)
                self.save_state(new_state)
                state = new_state

                if once:
                    return 1 if king_changed else 0

                if not king_changed:
                    log.info(
                        "Poll ok: king=%s config_version=%s (next in %.0fs)",
                        result.king_digest[:12],
                        result.config_version[:12] if result.config_version else "unknown",
                        self.poll_interval_sec,
                    )
                time.sleep(self.poll_interval_sec)
            except KeyboardInterrupt:
                log.info("Monitor stopped by user")
                return 0
            except (requests.RequestException, OSError, RuntimeError) as exc:
                log.error("Monitor poll failed: %s", exc)
                if once:
                    return 1
                time.sleep(self.poll_interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Teutonic dashboard for new kings and manifest changes.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help="Seconds between dashboard polls (default: 60)",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download the new king checkpoint when the digest changes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./king"),
        help="King download directory used with --auto-download",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="JSON file storing the last seen king digest and manifest version",
    )
    parser.add_argument("--dashboard-url", default=DASHBOARD_URL, help="Dashboard JSON URL")
    parser.add_argument("--manifest-url", default=DEFAULT_MANIFEST_URL, help="Dataset manifest URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--once", action="store_true", help="Poll once and exit (exit 1 if king changed)")
    parser.add_argument("--beep", action="store_true", help="Terminal beep on king or manifest change")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    monitor = DashboardMonitor(
        dashboard_url=args.dashboard_url,
        manifest_url=args.manifest_url,
        state_path=args.state_file,
        poll_interval_sec=args.interval,
        timeout=args.timeout,
        auto_download=args.auto_download,
        king_output_dir=args.output,
        beep_on_change=args.beep,
    )
    return monitor.run(once=args.once)


if __name__ == "__main__":
    sys.exit(main())
