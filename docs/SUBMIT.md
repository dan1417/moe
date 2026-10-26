# Submitting a challenger (one shot per hotkey — read fully first)

`teutonic-miner ready` is **irreversible**: it permanently consumes the hotkey and binds the uploaded
manifest on-chain. Every step below exists because a mistake there burns a hotkey (~1 TAO + the chance).

## 0. Decide with the parity eval (not with training loss)
```bash
source .venv-eval/bin/activate
export TEUTONIC_MODELS_DIRS=/opt/teutonic/models:/root/teutonic_models
python scripts/eval/parity_eval.py config --drift        # live config unchanged? upstream scoring code unchanged?
python scripts/eval/parity_eval.py duel --king reign:<current king> --challenger <recipe.json> \
       --sample tag:A --sample tag:B                     # FULL 30k each (no --limit)
```
Go only if, on **both** samples: `lcb > 0.003`, `robust_lcb > 0.003`, mu_hat ≥ ~0.0036, and no source
clearly negative. Re-check `king.reign_number` on https://teutonic.ai/dashboard.json — if the king changed,
re-run against the new king (your challenger must beat the king *at eval time*).

## 1. Materialize exactly what was scored
```bash
python scripts/merge/write_merge.py --recipe <recipe.json> --out /root/submit/<name>
python scripts/verify/verify_submission.py --model-dir /root/submit/<name> --king-dir <king dir>   # must print READY
python scripts/eval/parity_eval.py --limit 2000 duel --king reign:<king> --challenger /root/submit/<name> --sample tag:A
#   ^ the written directory must reproduce the recipe's mu_hat on the same 2000 sequences (± noise ~1e-4)
```
`verify_submission.py` checks: the 6 contract files vs live `chain.toml` SHA-256, no `manifest.json`,
no symlinks, ≤ 250 GB, tensor names/shapes/dtypes == king, all BF16, no `mtp` tensors, weights differ
from the king (validator digest), config lock keys.

## 2. Wallet (once)
```bash
python -m venv .venv-miner && source .venv-miner/bin/activate
git clone https://github.com/unarbos/teutonic.git /root/teutonic-miner && pip install -e '/root/teutonic-miner[miner]' bittensor-cli
btcli wallet new-coldkey --wallet-name miner
btcli wallet new-hotkey --wallet-name miner --hotkey h1 --crypto-type ed25519   # MUST be Ed25519
teutonic-miner configure --wallet-path "$HOME/.bittensor/wallets"
teutonic-miner register --wallet-name miner --hotkey-name h1 --network finney --netuid 3
```
Back up mnemonics offline. Registration burns `market.sn3_reg_burn_tao` (≈1 TAO today).

## 3. Upload, inspect, then ready
```bash
teutonic-miner auth --hotkey h1
teutonic-miner upload --hotkey h1 --name teutonic-II-110B-A7B-<suffix> /root/submit/<name>
teutonic-miner status                     # upload complete? manifest matches the directory?
teutonic-miner ready --hotkey h1          # IRREVERSIBLE — only after everything above passed
```
Then watch `queue` / `current_eval` / `history` in dashboard.json. Keep the winning hotkey registered:
it earns for the next ~4 reigns.
