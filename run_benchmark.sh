#!/usr/bin/env bash
#
# run_benchmark.sh — tentative federated-LoRA benchmark (CIFAR-100 / ViT)
#
# Runs all four methods (lalora, dp_lora, ffa_lora, rolora) over 3 seeds at a
# reduced 10 communication rounds, then plots mean+/-std accuracy curves.
#
# Usage, from the repository root on the GPU server:
#
#     bash run_benchmark.sh
#
# Notes
#   * Safe to re-run: existing per-run JSONs in results/ are overwritten.
#   * config.py is patched to 10 rounds ONLY for the duration of this script;
#     the trap below restores your original file on any exit (success, error,
#     or Ctrl-C).
#   * Assumes `uv` is the project's package manager (uv.lock is present). If the
#     server uses a plain virtualenv instead, replace `uv run` with your
#     interpreter and drop the `uv sync` step.

set -uo pipefail

# ---- configuration ----------------------------------------------------------
METHODS=(lalora dp_lora ffa_lora rolora)
SEEDS=(42 43 44)
ROUNDS=10
REQUIRED_TRANSFORMERS="4.46.3"   # 5.x silently random-inits the ViT backbone
SMOKE_MIN_ACC="0.02"             # random 100-class guessing is ~0.01
# -----------------------------------------------------------------------------

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

log() { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf '\n\033[31mABORT: %s\033[0m\n' "$*" >&2; exit 1; }

[ -f pyproject.toml ]                   || die "run from the repo root (pyproject.toml not found)"
[ -f federated_lora/config.py ]  || die "federated_lora/config.py not found"
command -v uv >/dev/null 2>&1    || die "uv not on PATH (see the note at the top of this script)"

# ---- 1. sync environment and assert the transformers pin --------------------
log "uv sync (installs the locked transformers==${REQUIRED_TRANSFORMERS}) ..."
uv sync || die "uv sync failed"

log "Verifying transformers version (guards the silent ViT random-init bug) ..."
uv run python - "$REQUIRED_TRANSFORMERS" <<'PY' || die "transformers is not the required version — fix the env before running"
import sys, transformers
required = sys.argv[1]
print("transformers:", transformers.__version__)
sys.exit(0 if transformers.__version__ == required else 3)
PY

log "Verifying CUDA is available ..."
uv run python - <<'PY' || die "CUDA not available — refusing to run this on CPU"
import sys, torch
ok = torch.cuda.is_available()
print("torch:", torch.__version__, "| cuda:", ok,
      "|", torch.cuda.get_device_name(0) if ok else "no gpu")
sys.exit(0 if ok else 4)
PY

# ---- 2. reversibly reduce the round count -----------------------------------
CONFIG="federated_lora/config.py"
BACKUP="${CONFIG}.benchmark.bak"

restore_config() {
	if [ -f "$BACKUP" ]; then
		mv -f "$BACKUP" "$CONFIG"
		log "Restored original ${CONFIG}"
	fi
}
trap restore_config EXIT INT TERM
cp "$CONFIG" "$BACKUP"

set_rounds() {
	local n="$1"
	sed -i -E "s/(global_rounds: int = )[0-9]+/\1${n}/" "$CONFIG"
	local got
	got="$(uv run python -c 'from federated_lora.config import Config; print(Config().global_rounds)')"
	[ "$got" = "$n" ] || die "could not set global_rounds=${n} (config reports ${got})"
}

# ---- 3. smoke test: 1 round, confirm the backbone actually loaded -----------
log "Smoke test (1 round of lalora) to confirm the pipeline is valid post-fix ..."
set_rounds 1
mkdir -p logs results_smoke
smoke_log="logs/smoke.log"
if uv run python -m federated_lora run --method lalora --seed 42 --results-dir results_smoke 2>&1 | tee "$smoke_log"; then
	acc="$(grep -oE 'acc=[0-9.]+' "$smoke_log" | tail -1 | cut -d= -f2)"
	log "Smoke accuracy after 1 round: ${acc:-<none>}"
	awk "BEGIN{exit !((${acc:-0}) >= ${SMOKE_MIN_ACC})}" \
		|| die "smoke accuracy ${acc:-0} < ${SMOKE_MIN_ACC}: backbone likely NOT loaded (transformers bug?). Aborting before the long run."
else
	die "smoke run crashed — see ${smoke_log}"
fi

# ---- 4. the benchmark -------------------------------------------------------
set_rounds "$ROUNDS"
mkdir -p results figures logs
log "Benchmark: ${#METHODS[@]} methods x ${#SEEDS[@]} seeds @ ${ROUNDS} rounds"
overall_start=$(date +%s)
declare -a SUMMARY=()

for m in "${METHODS[@]}"; do
	for s in "${SEEDS[@]}"; do
		run_log="logs/${m}_seed${s}.log"
		log "RUN method=${m} seed=${s} -> ${run_log}"
		t0=$(date +%s)
		if uv run python -m federated_lora run --method "$m" --seed "$s" 2>&1 | tee "$run_log"; then
			status="ok"
		else
			status="FAILED"
		fi
		t1=$(date +%s)
		acc="$(grep -oE 'acc=[0-9.]+' "$run_log" | tail -1 | cut -d= -f2)"
		SUMMARY+=("$(printf '%-9s seed=%-2s %-7s final_acc=%-8s %4dm' \
			"$m" "$s" "$status" "${acc:-n/a}" $(((t1 - t0) / 60)))")
	done
done

# ---- 5. plot ----------------------------------------------------------------
log "Generating figures ..."
uv run python -m federated_lora plot results/ --output-dir figures 2>&1 | tee logs/plot.log \
	|| log "WARN: plotting failed — the per-run JSONs are still saved in results/"

# ---- 6. summary -------------------------------------------------------------
overall_end=$(date +%s)
log "DONE in $(((overall_end - overall_start) / 60)) min. Summary:"
printf '  %s\n' "${SUMMARY[@]}"
echo
echo "  Results JSON : $(pwd)/results/"
echo "  Figures      : $(pwd)/figures/"
echo "  Per-run logs : $(pwd)/logs/"
