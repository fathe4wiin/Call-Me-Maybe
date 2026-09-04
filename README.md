*This project has been created as part of the 42 curriculum by fathe4wiin.*

# Call Me Maybe

## Description

Call Me Maybe turns a natural-language request into a structured function
call. Given a prompt such as *"What is the sum of 40 and 2?"*, the program
does not answer `42`. It returns the tool name and typed arguments, for
example `fn_add_numbers` with `{"a": 40, "b": 2}`.

Small models are unreliable at emitting JSON on their own. This project uses
**constrained decoding**: at every generation step the logits of tokens that
would break JSON syntax or the catalog schema are set to negative infinity.
The model still chooses *which* function to call and *which* values to fill
in; the decoder only forbids illegal structure. The result is always
parseable JSON that matches `functions_definition.json`.

The default model is `Qwen/Qwen3-0.6B`, accessed only through the public
`llm_sdk` API (`encode`, `get_logits_from_input_ids`, vocab path helpers).

## Instructions

Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) are required. The reviewer
runs `uv sync`. The SDK lives in `llm_sdk/` next to `src/`.

```bash
make install    # uv sync
make run        # uv run python -m src
make lint       # flake8 and mypy with the subject flags
make debug      # pdb on src/__main__.py
make clean      # __pycache__ and .mypy_cache
```

Default paths are `data/input/` for the catalog and prompts, and
`data/output/function_calling_results.json` for results. Override them with:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The first run downloads Qwen3-0.6B (~1.5 GB) from the Hugging Face Hub.

## Example usage

```bash
uv sync
uv run python -m src
cat data/output/function_calling_results.json
```

Expected shape (keys exactly `prompt`, `name`, `parameters`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

Missing or invalid input files print one `error: <path>: <reason>` line on
stderr and exit with status 1.

## Algorithm explanation

The target JSON is a **fixed template** with holes for the function name and
the argument values:

```text
{"name":"<FUNCTION>","parameters":{"<key>":<value>, ...}}
```

Everything else (braces, quotes, keys, colons, commas) is forced. Generation
is a loop over `get_logits_from_input_ids`:

1. Encode the steering prompt plus the prefix `{"name":"`.
2. Ask the model for next-token logits.
3. Map every vocabulary id to text using `get_path_to_vocab_file()` (byte-level
   BPE: `Ġ` is a space).
4. Ask the constraint machine which tokens are still legal.
5. Copy logits into a NumPy array, set illegal entries to `-inf`, take argmax.
6. Append that token, advance the machine, repeat until the JSON object is
   closed.

The constraint machine is character-level. Phases:

- **NAME** — only prefixes of catalog function names, then `"`.
- **LITERAL** — the next exact structural fragment
  (`,"parameters":{`, `"key":`, `}}`, …).
- **VALUE** — `string`, `number` / `integer`, or `boolean` according to the
  chosen function’s schema. JSON numbers allow an optional minus, digits, and
  (for `number`) a fractional part. Strings allow escaped quotes. Booleans
  are only `true` / `false`.
- **DONE** — stop; do not wait for an EOS token.

A multi-character token is accepted only if every character in order is legal
(the machine is snapshotted and restored while testing). After a finished
number or boolean, `,` or `}` is re-dispatched into the following literal so
a token such as `2}` can both end the value and close the object.

The prompt lists function names, descriptions, and parameter types so the
model can pick among legal options. The prompt is not trusted to produce
JSON by itself.

## Design decisions

- **Character rules, token masking.** Validity is “which characters may come
  next.” The vocabulary module turns that into token ids. The two layers stay
  small.
- **Canonical compact JSON.** No optional whitespace in the generated object.
  That shrinks the set of legal tokens at structural positions to a handful.
- **Forced keys, free values.** After the model picks a function name, key
  order follows the catalog. The LLM fills values; it does not invent keys.
- **Public SDK only.** No `_model`, `_tokenizer`, `torch`, `transformers`, or
  Hugging Face imports in `src/`. `encode` is converted with `.tolist()` so
  the tensor type never needs to be named.
- **Pydantic everywhere.** Catalog, prompts, CLI paths, vocab tables, the
  decoder state, and output records are pydantic models.
- **One error style.** File and JSON problems print `error: <path>: ...` and
  exit 1, with no traceback.

## Performance analysis

Constrained decoding guarantees **100% valid, schema-shaped JSON** by
construction: illegal tokens cannot be selected, and generation stops when
the object is closed. Function choice and free-text / regex values still
depend on Qwen3-0.6B; the subject target is 90%+ correct selection and
argument extraction on the evaluation set.

Speed: each step scores ~150k logits but only tests tokens whose first
character is still allowed, except inside an open string. Structural steps
often keep a few tokens. The full sample file should stay under the
5-minute budget after the one-time model download. This checkout was not
run against the LLM (no model weights on the machine used to write the
code).

## Challenges faced

- **Byte-level BPE.** `vocab.json` does not store plain UTF-8. Reconstructing
  GPT-2 `bytes_to_unicode` is required so `Ġ` becomes a space and punctuation
  tokens match the JSON we are emitting.
- **Ending a number.** JSON numbers have no closer. Once the number is
  valid, `,` or `}` is offered; when the model picks it, that character is
  fed into the next literal.
- **Tokens that span a phase change.** A single token may contain the last
  digit of a number and the following `}`. `accepts()` walks every character
  with snapshot/restore so those tokens stay legal when they should.
- **Tokenizer file variants.** The SDK prefers `vocab.json`; if that path
  fails, `get_path_to_tokenizer_file()` (`tokenizer.json`) is used instead.

## Testing strategy

The implementation is built to be checked as follows (without hardcoding the
sample prompts):

- Point `--input` or `--functions_definition` at a missing file or at the
  wrong JSON: one stderr line, exit 1.
- Run `uv run python -m src` on the provided files and confirm the output
  array length equals the prompt count, each object has exactly `prompt`,
  `name`, `parameters`, names exist in the catalog, and types match.
- Swap in another catalog (boolean, integer, extra parameters) and confirm
  the decoder still emits valid JSON.
- `make lint` for flake8 and the required mypy flags.

Unit tests are local-only and not part of the submission.

## Resources

- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Tokenizer summary (byte-level BPE)](https://huggingface.co/docs/transformers/en/tokenizer_summary)

### Use of AI

AI was used as a coding assistant while reading `en.subject.pdf`, implementing
constrained decoding in `src/`, wiring the Makefile / pydantic I/O, and
writing this README and `report.md`. Generated code was checked against the
subject (Python 3.10+, flake8/mypy, pydantic-only classes, numpy/json only,
no private `llm_sdk` attributes, LLM chooses the function, 100% valid JSON).
The program was not executed on the authoring machine because the model
runtime was not available there.
