#!/usr/bin/env bash
# Prepare private then public moulinette sets, generate answers, and grade
# each set as soon as generation finishes (do not wait until the end).

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOULINETTE_DIR="$ROOT/moulinette"
DATA_DIR="$ROOT/data"
OUTPUT_DIR="$DATA_DIR/output"

export PYTHONUNBUFFERED=1

banner() {
  echo
  echo "=============================================================================="
  echo " $*"
  echo "=============================================================================="
}

format_duration() {
  local total_seconds="$1"
  printf '%d:%02d' "$((total_seconds / 60))" "$((total_seconds % 60))"
}

run_set() {
  local set_name="$1"
  local output_file="$OUTPUT_DIR/function_calling_results_${set_name}.json"
  local set_start generate_start generate_end set_end
  local generate_seconds set_seconds

  set_start=$(date +%s)

  banner "PREPARE ${set_name} exercises"
  (
    cd "$MOULINETTE_DIR"
    uv run python -m moulinette prepare_exercises --set "$set_name" --output "$DATA_DIR"
  ) || {
    echo "error: prepare_exercises failed for ${set_name}" >&2
    return 1
  }

  mkdir -p "$OUTPUT_DIR"

  banner "GENERATE ${set_name} student answers -> ${output_file}"
  generate_start=$(date +%s)
  (
    cd "$ROOT"
    uv run python -m src \
      --functions_definition "$DATA_DIR/input/functions_definition.json" \
      --input "$DATA_DIR/input/function_calling_tests.json" \
      --output "$output_file"
  ) || {
    echo "error: generation failed for ${set_name}" >&2
    return 1
  }
  generate_end=$(date +%s)
  generate_seconds=$((generate_end - generate_start))
  echo
  echo "TIME ${set_name} generation: $(format_duration "$generate_seconds") (${generate_seconds}s)"
  if [ "$generate_seconds" -lt 300 ]; then
    echo "TIME ${set_name} generation: UNDER 5:00"
  else
    echo "TIME ${set_name} generation: OVER 5:00"
  fi

  banner "MOULINETTE ${set_name} results (printed now)"
  (
    cd "$MOULINETTE_DIR"
    uv run python -m moulinette grade_student_answers \
      --set "$set_name" \
      --student_answer_path "$output_file"
  )
  set_end=$(date +%s)
  set_seconds=$((set_end - set_start))
  echo "TIME ${set_name} full set (prepare + generate + grade): $(format_duration "$set_seconds") (${set_seconds}s)"
}

banner "uv sync (project)"
(cd "$ROOT" && uv sync) || exit 1

banner "uv sync (moulinette)"
(cd "$MOULINETTE_DIR" && uv sync) || exit 1

run_set private
private_status=$?

run_set public
public_status=$?

banner "DONE (private exit=${private_status}, public exit=${public_status})"
if [ "$private_status" -ne 0 ] || [ "$public_status" -ne 0 ]; then
  exit 1
fi
