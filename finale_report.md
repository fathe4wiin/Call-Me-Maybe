# Call Me Maybe — finale report

Lint configs and Makefile flags were left unchanged.
`make lint` now exits 0: flake8 is clean, mypy reports
`Success: no issues found in 16 source files`.

## 1. Lint work (code only)

`.flake8` (`max-line-length = 100`) and the Makefile mypy flags
(`--warn-return-any --warn-unused-ignores --ignore-missing-imports
--disallow-untyped-defs --check-untyped-defs`) were not edited.

### flake8

| File | Rule | Fix |
|---|---|---|
| `src/constraints.py` | E261 / E262 / E501 | Removed the over-long inline comment on `assert_never` |
| `moulinette/moulinette/extract_functions_infos.py` | F401 | Dropped unused `List` |
| `moulinette/moulinette/generate_tests_and_corrections.py` | F401 | Dropped unused `exercises` import |
| `moulinette/moulinette/output_formatter.py` | E302 | Blank line before `_supports_color` |
| `moulinette/moulinette/__main__.py` | E501 | Split the expected-call print |
| `moulinette/moulinette/functions_definition.py` | E501 | Wrapped two long prompt strings (same text) |

### mypy

| File | Rule | Fix |
|---|---|---|
| `src/generate.py` | `attr-defined` | `llm_sdk` is excluded from mypy, so `from llm_sdk import Small_LLM_Model` failed. The public constructor is now loaded with `getattr` and typed through a `Protocol` |
| `moulinette/moulinette/__main__.py` | `no-untyped-def` | `Moulinette.__init__` → `-> None` |
| `moulinette/moulinette/output_formatter.py` | `no-untyped-def` | `ColoredOutput.__init__` → `-> None` |

`src/load.py` also got a PEP 257 docstring on `_load_list`. That was not a
lint failure.

## 2. Subject checklist (v1.5)

Mandatory items that **are** in place:

| Requirement | Status |
|---|---|
| Python 3.10+, `uv sync`, `uv.lock` | Yes (`requires-python = ">=3.10"`) |
| Entry `uv run python -m src` / `make run` | Yes |
| Flags `--functions_definition`, `--input`, `--output` with the stated defaults | Yes (`src/cli.py`) |
| Pydantic catalog / prompt / output models | Yes (`src/models.py`) |
| Graceful JSON / path errors: one stderr line, exit 1, no traceback | Yes (`src/load.py`, save / decode wrappers) |
| Empty catalog → error + exit 1 | Yes |
| Function chosen by the LLM, not keyword `if`/`else` | Yes |
| Constrained decoding: illegal logits = `-inf`, then argmax | Yes (`src/generate.py`) |
| Compact JSON template, forced keys, typed values | Yes (`src/constraints.py`) |
| `number` coerced to `float` so output writes `2.0` not `2` | Yes (`coerce_parameters`) |
| `integer` / `boolean` / `string` value machines | Yes |
| Public SDK only (`encode`, logits, vocab / tokenizer paths) | Yes; no `_model` / `_tokenizer` / `torch` in `src/` |
| Allowed extras only (`numpy`, `json`, stdlib, pydantic) | Yes |
| Output keys exactly `prompt`, `name`, `parameters` | Yes |
| Makefile `install` / `run` / `debug` / `clean` / `lint` | Yes |
| `.gitignore` has `.venv`, caches, `data/output/` | Yes |
| Required tree: `src/`, `pyproject.toml`, `uv.lock`, `llm_sdk/`, `data/input/`, `README.md` | Yes |
| Type hints + module / public docstrings in `src/` | Yes (see leftover package below) |

## 3. Missing or weak points

These are the gaps a reviewer can still hit. Lint is not one of them anymore.

### Must-fix or high risk

1. **The pipeline has not been run against Qwen on this machine.**
   The README says the same. Subject V.5 still asks for **90%+ correct
   calls**, **100% valid JSON**, and **under 5 minutes** after the one-time
   download. The mask can guarantee valid JSON. Accuracy and speed are
   unproven until `make run` finishes on the real model.

2. **`data/input/` currently holds the private grading set**, not the
   public sample the docs describe.
   - Files now: `fn_multiply_numbers`, `fn_is_even`,
     `fn_calculate_compound_interest`, `fn_execute_sql_query`,
     `fn_read_file`, `fn_format_template` (11 private prompts).
   - Docs / walkthrough still talk about `fn_add_numbers`, `fn_greet`,
     `fn_reverse_string`, `fn_get_square_root`,
     `fn_substitute_string_with_regex`.
   - If this repo is submitted as-is, the private moulinette prompts sit
     in git. Reviewers inject their own JSON, so the decoder can still
     work, but the committed sample is the wrong set.

3. **Leftover uv package `src/call_me_maybe/`.**
   It only prints `Hello from call-me-maybe!`.
   `pyproject.toml` still has
   `call-me-maybe = "call_me_maybe:main"`.
   The subject entry is `python -m src`. Running `uv run call-me-maybe`
   is the wrong program. No module / function docstring there (PEP 257).

### Strict subject reading (probably not graded, but visible)

4. **“All classes are pydantic models.”**
   Business classes are pydantic (`CliArgs`, `JsonLoader`, catalog /
   output models, `Vocabulary`, `DecodeState`).
   These are **not** pydantic:
   - `Phase`, `ParamKind`, `AfterLiteral` (`enum.Enum`)
   - `SmallLLMModel` (`typing.Protocol`, added so mypy can type the SDK)
   Enums / Protocol are the normal way to write this. A staff member who
   greps for `class` and rejects anything that is not `BaseModel` could
   still complain.

5. **Quality bar not demonstrated.**
   No checked-in `data/output/` (correct: it is gitignored). There is also
   no local run log that the 11-prompt file finishes in time or scores
   90%+.

### Optional / bonus (subject VII) — not implemented

- Other models besides Qwen3-0.6B
- Recoded tokenize / detokenize that avoids `encode` / `decode`
- Caching / batching
- Graded test suite
- Generation visualizer
- Nested argument objects

These are optional. Do not claim them.

### Extra files (not required, not forbidden)

`moulinette/`, `project.md`, `walkthrough.md`, `steps.md`, `report.md`,
`todo.md`, `Untitled`, and similar notes are outside the required
submission set. `Untitled` is leftover scratch (`masked[valid] = scores[valid]`).
`moulinette/` was only touched so `make lint` (which runs on `.`) stays
green; grading behavior was not changed.

## 4. What to do before a defense

1. Restore **public** files under `data/input/` if this clone is what you
   submit, so the private set is not in git.
2. Run `make install` then `make run` once the ~1.5 GB weights are
   available. Confirm 11 (or N) output objects, float `number` values,
   and runtime under 5 minutes.
3. Re-check a broken path:
   `uv run python -m src --input /tmp/missing.json`
   → one stderr line, exit 1.
4. Remove or ignore `src/call_me_maybe/` and the `call-me-maybe` script
   if you want the console entry to match the subject.
5. Delete `Untitled` if you do not want junk in the tree.

## 5. Bottom line

- **Lint:** fixed in code; config flags untouched; `make lint` passes.
- **Mandatory pipeline:** implemented (parse → constrain → coerce → write).
- **Open subject gaps:** no live LLM run, input JSON is the private set,
  leftover `call_me_maybe` hello-world package.
