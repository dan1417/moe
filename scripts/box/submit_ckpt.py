"""Submit a staged checkpoint to a Teutonic competition with a fresh hotkey (lessons of sn3auto001/003 built in).

  .venv-miner/bin/python /data/teutonic/logs/submit_ckpt.py --hotkey sn3auto004 --model-dir /data/teutonic/submit/c1_s150 \
      --name teutonic-II-110B-A7B-004 --competition text [--ready]

prepare (default): Ed25519 check -> registration check (saves miner state) -> verify_submission READY -> upload auth ->
empty the registration prefix and confirm it (stale objects / multipart parts count toward the 250 GB wipe limit) ->
upload ONCE with 8 files x 8 part streams (16x16 dropped SSL connections on this box) -> prefix_check PASS.
--ready: additionally commits the ready signal after the king re-check (IRREVERSIBLE).
Stops at the first failure.
"""
import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path("/root/workspace/MoE")
MINER = str(REPO / ".venv-miner/bin/teutonic-miner")
WALLET = "autotao"
WALLET_PATH = Path.home() / ".bittensor" / "wallets"
KING_DIGEST = {"text": "8864c279", "main": "9b202f1a", "code": "fa428e5e"}  # kings the candidates were scored against (2026-10-10)


def run(*cmd, check=True):
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True)
    print((proc.stdout + proc.stderr).strip()[-3000:], flush=True)
    if check and proc.returncode != 0:
        sys.exit(f"FAILED ({proc.returncode}): {' '.join(cmd)}")
    return proc.stdout + proc.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotkey", required=True)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--competition", default="text", choices=("main", "math", "code", "text"))
    ap.add_argument("--ready", action="store_true", help="commit the ready signal after a successful prepare (IRREVERSIBLE)")
    args = ap.parse_args()

    print("== Ed25519 hotkey", flush=True)
    run(MINER, "check", "--wallet-name", WALLET, "--hotkey-name", args.hotkey)
    print("== registration (check only)", flush=True)
    out = run(MINER, "register", "--wallet-name", WALLET, "--hotkey-name", args.hotkey, "--network", "finney", "--netuid", "3", "--check-only")
    state_dir = Path(next(tok.split("=", 1)[1] for tok in out.split() if tok.startswith("state_dir=")))
    run(MINER, "use", args.hotkey)  # several saved miners on this box: make this one active for status/auth/upload
    status = run(MINER, "status")
    if "eligibility=available" not in status:
        sys.exit("hotkey not available (already consumed?)")
    print("== submission directory", flush=True)
    king_dir = "/data/teutonic/models/95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0"
    out = run(str(REPO / ".venv-eval/bin/python"), "scripts/verify/verify_submission.py", "--model-dir", str(args.model_dir), "--king-dir", king_dir)
    if "READY: 0 failure" not in out:
        sys.exit("verify_submission did not print READY")
    print("== upload authorization", flush=True)
    run(MINER, "auth", "--hotkey", args.hotkey)

    if not (state_dir / "manifest.json").exists():
        import boto3
        from botocore.config import Config
        auth = json.loads((state_dir / "upload-auth.json").read_text())
        s3 = boto3.client("s3", endpoint_url=auth["r2_endpoint"], aws_access_key_id=auth["access_key_id"],
                          aws_secret_access_key=auth["secret_access_key"], aws_session_token=auth["session_token"],
                          region_name="auto", config=Config(signature_version="s3v4", retries={"max_attempts": 10}))
        bucket, prefix = auth["private_model_bucket"], auth["allowed_prefix"]
        listed = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
        for o in listed:
            s3.delete_object(Bucket=bucket, Key=o["Key"])
        for u in s3.list_multipart_uploads(Bucket=bucket, Prefix=prefix).get("Uploads", []):
            s3.abort_multipart_upload(Bucket=bucket, Key=u["Key"], UploadId=u["UploadId"])
        left = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("KeyCount", 0)
        parts = len(s3.list_multipart_uploads(Bucket=bucket, Prefix=prefix).get("Uploads", []))
        print(f"== prefix emptied ({len(listed)} stale objects removed): {left} objects, {parts} multipart uploads", flush=True)
        if left or parts:
            sys.exit("prefix not empty -- stop and inspect")
        print(f"== upload {args.model_dir} as {args.name} (once, 8x8 streams)", flush=True)
        from miner import upload_model
        upload_model.FILE_WORKERS = 8
        upload_model.PART_STREAMS = 8
        rc = upload_model.main(["--wallet-name", WALLET, "--hotkey-name", args.hotkey, "--wallet-path", str(WALLET_PATH),
                                "--state-dir", str(state_dir), "--model-dir", str(args.model_dir), "--model-name", args.name])
        if rc:
            sys.exit(f"upload failed ({rc}) -- inspect the prefix before anything else")
    else:
        print("== manifest already present: upload done before, not uploading again", flush=True)
    print("== prefix check", flush=True)
    out = run(str(REPO / ".venv-miner/bin/python"), "scripts/submit/prefix_check.py", "--hotkey", state_dir.name)
    if "PASS" not in out:
        sys.exit("prefix check failed -- do NOT ready")
    print("PREPARED", flush=True)
    if not args.ready:
        return
    print("== king re-check", flush=True)
    url = "https://teutonic.ai/dashboard.json" if args.competition == "main" else f"https://teutonic.ai/competitions/{args.competition}/dashboard.json"
    d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "teutonic-miner/1.0"}), timeout=60).read())
    king = (d.get("king") or {})
    print("king", king.get("model_repo"), (king.get("king_digest") or "")[:12], "queue", len(d.get("queue") or []), flush=True)
    if args.competition in KING_DIGEST and not (king.get("king_digest") or "").startswith(KING_DIGEST[args.competition]):
        sys.exit("the king changed since the checkpoint was scored -- re-score before ready")
    print(f"== ready --competition {args.competition} (IRREVERSIBLE)", flush=True)
    run(MINER, "ready", "--hotkey", args.hotkey, "--competition", args.competition)
    print("READY COMMITTED", flush=True)


if __name__ == "__main__":
    main()
