# Call Me Maybe — work list

Work in `src/`, not `main.py`. The school runs `uv run python -m src`.
`main.py` is only a sandbox (untracked or delete it).

Target layout:

```text
src/
  __init__.py      # package marker
  __main__.py      # this is what `python -m src` runs
  cli.py           # argparse / flags
  models.py        # pydantic models
  load.py          # read + validate JSON
  ...              # decoder later
data/input/        # already there — read these
data/output/       # create at runtime; do not git-add results
```

Ignore `src/call_me_maybe/` (uv default). That is not the subject entry point.

---

## Phase 1 — parsing (do this first, no LLM)

- [ ] Confirm `src/` is a package: from the venv, `python -m src` runs (even if it only hits `pass`).
- [ ] CLI in `cli.py`, called from `__main__.py`
  - Flags: `--functions_definition`, `--input`, `--output`
  - Defaults: `data/input/` and `data/output/`
  - The PDF names the default output file two ways (`function_calling_results.json` vs `function_calls.json`). Pick one when `--output` is omitted. `--output` always wins when given.
- [ ] Pydantic models in `models.py` (every class you write must be pydantic)
  - Function catalog: `name`, `description`, `parameters` (map of name → `{type}`), `returns`
  - Tests: list of objects with `prompt`
  - Output record: `prompt`, `name`, `parameters` (fill later)
- [ ] Load + validate in `load.py`
  - Open files with a context manager, `json.load`, then pydantic
  - Missing file / unreadable / invalid JSON: clear error, exit, no traceback
- [ ] Sanity check, still no model
  - Print how many functions and how many prompts loaded
  - Confirm parameter types (`number` vs `string`)
  - Do **not** call `Small_LLM_Model()` yet
  - Parser must follow the schema, not the sample function names (reviewers swap the JSON)

Parsing is done when `python -m src` loads both JSON files through pydantic and fails cleanly on a broken file.

---

## Phase 2 — project wiring (after parsing works)

- [ ] Makefile `run` → `uv run python -m src` (same flags)
- [ ] Makefile `install`, `debug`, `clean`, `lint`
- [ ] `.gitignore`: `.venv`, `__pycache__`, `.mypy_cache`, `data/output/`
- [ ] Commit `uv.lock`; do not commit generated output
- [ ] `uv add --dev flake8 mypy` and a `lint` target with the mypy flags from the subject
- [ ] Pin `requires-python = ">=3.10"` in `pyproject.toml` (not `>=3.14`)

---

## Phase 3 — constrained decoding (separate, after 1–2)

- [ ] Build a prompt (user question + enough catalog so the model can pick a function)
- [ ] `encode` it
- [ ] Ask for logits
- [ ] Mask token IDs that would be illegal right now (JSON broken, or type/schema broken)
- [ ] Pick among the rest (usually the highest remaining logit)
- [ ] Append that ID; repeat until the JSON object is finished
- [ ] Write the output JSON array (exact keys, types matching the catalog)

---

## Do not

- Import `torch`, `transformers`, or Hugging Face packages in `src/`
- Touch private `llm_sdk` attributes (`_model`, `_tokenizer`, …)
- Pick the function with if/else on keywords — the LLM must choose
- Hardcode answers from the sample files
