#!/bin/bash
set -euo pipefail

MISSION="${1:?usage: $0 MISSION [OUTPUT_ROOT]}"
OUTPUT_ROOT="${2:-${GRANDTOUR_ROOT:?set GRANDTOUR_ROOT or pass OUTPUT_ROOT}}"

mkdir -p "$OUTPUT_ROOT"

# The dataset is gated. Authenticate first with `hf auth login` and accept the
# dataset terms in the Hub UI. Download one mission before scaling up.
hf download leggedrobotics/grand_tour_dataset \
  --type dataset \
  --include "${MISSION}/*" \
  --local-dir "$OUTPUT_ROOT"

# Hugging Face serves the mission as topic-level tar archives.  Keep the
# downloaded archive only when explicitly requested; training reads the
# extracted mission layout.
uv run quadwm grandtour extract --root "$OUTPUT_ROOT" --mission "$MISSION"
