# Resolve MoE repo root from any nested scripts/* path.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/../_repo_root.sh"  OR from scripts/: source _repo_root.sh
# Sets REPO_ROOT (absolute).
_moe_find_root() {
  local start="${1:-${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}}"
  local dir
  dir="$(cd "$(dirname "$start")" && pwd)"
  while [[ "$dir" != "/" ]]; do
    if [[ -d "$dir/.git" || -f "$dir/requirements.txt" && -d "$dir/src" ]]; then
      printf '%s' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(_moe_find_root "${BASH_SOURCE[0]}")"
  export REPO_ROOT
fi
