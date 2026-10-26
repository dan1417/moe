#!/usr/bin/env python3
"""Print campaign43 king arguments for the reigning kings of the given competitions (default main,math):

    king_args.py main,math    ->  --king main=M45:0.003 --king math=MA2:0.002

A thin wrapper over scripts/box/kings.py (live dashboard digest -> public_models.json name -> ingested model in
models.json). A king that does not resolve (not ingested yet, dashboard unreachable, unknown competition) prints nothing
on stdout, the reason on stderr, and exits 2: the caller skips the screen instead of scoring against a guessed king (this
script used to fall back to K43, which is not in models.json, and the vacant MATH crown to the MAIN king).
main_watch.sh runs the deployed copy /data/teutonic/logs/king_args.py; it imports kings.py from the repo.
"""
import sys
from pathlib import Path

REPO_BOX = Path("/root/workspace/MoE/scripts/box")
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here if (_here / "kings.py").exists() and _here.name == "box" else REPO_BOX))


def main() -> int:
    comps = [c for c in (sys.argv[1] if len(sys.argv) > 1 else "main,math").split(",") if c]
    try:
        import kings
        unknown = [c for c in comps if c not in kings.COMPS]
        if unknown or not comps:
            print(f"king_args: unknown competition {','.join(unknown) or '(none)'} (known: {','.join(kings.COMPS)})",
                  file=sys.stderr)
            return 2
        out = kings.king_args(comps)
    except Exception as exc:  # noqa: BLE001 - KingNotReady or anything else: no arguments, the screen is skipped
        print(f"king_args: KING NOT READY: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
