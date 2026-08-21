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

## Phase 1 — parsing (done)

- [x] Confirm `src/` is a package: from the venv, `python -m src` runs
- [x] CLI in `cli.py`, called from `__main__.py`
  - Flags: `--functions_definition`, `--input`, `--output`
  - Defaults: `data/input/functions_definition.json`, `data/input/function_calling_tests.json`, `data/output/function_calling_results.json`
- [x] Pydantic models in `models.py`
- [x] Load + validate in `load.py` (missing / invalid JSON → one error line, exit 1)
- [x] Sanity check prints function signatures and prompt count — still no LLM

Check it: `uv run python -m src` and `uv run python -m src --help`.
Break it: point `--input` at a missing file or at the catalog JSON.

---

## Next — first LLM contact (do this before the full decoder)

This is the step after parsing. Still **not** the whole project.

- [ ] In a scratch script or a new `src/` module, construct `Small_LLM_Model()` once (first run downloads Qwen)
- [ ] `encode` a short string; print the token ids; `decode` them back
- [ ] Turn those ids into a `list[int]` (encode returns a 2-D tensor) and call `get_logits_from_input_ids`
- [ ] Find the index with the highest logit; `decode` that single token
- [ ] Open `get_path_to_vocab_file()` (and if needed `get_path_to_tokenizer_file()`) and look at how `{`, `"`, a digit, and a space are stored

Goal: you can explain “prompt → ids → logits → next token” out loud. Then start Phase 3 (masking).

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
