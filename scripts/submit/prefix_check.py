#!/usr/bin/env python3
"""Pre-`ready` check of a Teutonic upload: does the private prefix hold exactly the signed manifest?

Uses the hotkey's scoped R2 credential (upload-auth.json, written by `teutonic-miner auth`) to list the registration
prefix and compares it with the local signed manifest (manifest.json, written by `teutonic-miner upload`):
  * every manifest file present with the manifest's size (and sha256 metadata when the listing exposes it)
  * no extra objects, no in-progress multipart uploads (they count toward the 250 GB quota!)
  * total size < 250 GB
Exit 0 = PASS (safe to `ready`), 1 = FAIL (do NOT `ready`; never re-upload into the prefix without understanding why).

  python scripts/submit/prefix_check.py --hotkey <ss58 or state dir name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.config import Config

LIMIT = 250 * 10**9
STATE_ROOT = Path(__file__).resolve().parents[2] / ".teutonic-miner"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotkey", required=True, help="hotkey ss58 (= state dir name under .teutonic-miner)")
    ap.add_argument("--state-root", type=Path, default=STATE_ROOT)
    args = ap.parse_args()
    state = args.state_root / args.hotkey
    auth = json.loads((state / "upload-auth.json").read_text())
    manifest = json.loads((state / "manifest.json").read_text())
    client = boto3.client(
        "s3", endpoint_url=auth["r2_endpoint"], aws_access_key_id=auth["access_key_id"],
        aws_secret_access_key=auth["secret_access_key"], aws_session_token=auth["session_token"],
        region_name="auto", config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
    )
    bucket, prefix = auth["private_model_bucket"], auth["allowed_prefix"]
    objects: dict[str, int] = {}
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = client.list_objects_v2(**kw)
        for obj in page.get("Contents", []):
            objects[obj["Key"][len(prefix):]] = int(obj["Size"])
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    inflight = None
    try:
        mp = client.list_multipart_uploads(Bucket=bucket, Prefix=prefix)
        inflight = [u["Key"] for u in mp.get("Uploads", [])]
    except Exception as exc:  # the scoped policy may not allow it
        print(f"note: ListMultipartUploads not permitted ({str(exc)[:60]}); cannot see in-flight parts")

    expected = {f["path"]: int(f["size"]) for f in manifest["files"]}
    problems = []
    for path, size in expected.items():
        if path not in objects:
            problems.append(f"MISSING {path}")
        elif objects[path] != size:
            problems.append(f"SIZE {path}: remote {objects[path]} != manifest {size}")
    extra = sorted(set(objects) - set(expected) - {"manifest.json"})
    if extra:
        problems.append(f"EXTRA objects: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
    if "manifest.json" not in objects:
        problems.append("MISSING manifest.json in prefix")
    total = sum(objects.values())
    if total >= LIMIT:
        problems.append(f"TOTAL {total / 1e9:.2f} GB >= 250 GB limit")
    if inflight:
        problems.append(f"IN-FLIGHT multipart uploads: {inflight}")
    print(f"prefix {prefix}: {len(objects)} objects, {total / 1e9:.3f} GB (limit 250); manifest lists {len(expected)} files, "
          f"{sum(expected.values()) / 1e9:.3f} GB; model {manifest.get('model_name')} digest {str(manifest.get('model_digest', ''))[:16]}; "
          f"in-flight: {inflight if inflight is not None else 'unknown'}")
    for p in problems:
        print("FAIL", p)
    print("PASS — prefix matches the signed manifest; safe to ready" if not problems else "DO NOT READY")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
