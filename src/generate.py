"""Constrained decoding: mask illegal tokens until the JSON object is closed."""

from __future__ import annotations

import json
import sys

import numpy as np
from llm_sdk import Small_LLM_Model

from .constraints import DecodeState
from .models import FunctionDefinition, OutputRecord, TestPrompt
from .prompt import build_generation_prompt
from .vocabulary import Vocabulary

_MAX_NEW_TOKENS = 256
_EMERGENCY_CHARS = 512


def tensor_to_ids(encoded: object) -> list[int]:
    """Flatten the 2-D tensor from ``encode`` into a list of token ids.

    Args:
        encoded: Return value of ``Small_LLM_Model.encode``.

    Returns:
        Token ids for a single sequence.

    Raises:
        TypeError: If the object is not tensor-like.
    """
    to_list = getattr(encoded, "tolist", None)
    if not callable(to_list):
        raise TypeError("encode() result does not support tolist()")
    raw: object = to_list()
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        return [int(item) for item in raw[0]]
    if isinstance(raw, list):
        return [int(item) for item in raw]
    raise TypeError("encode() returned an unexpected shape")


def load_model() -> Small_LLM_Model:
    """Construct Qwen3-0.6B once. First call downloads the weights."""
    print(
        "loading Qwen/Qwen3-0.6B (first run downloads weights)...",
        flush=True,
    )
    return Small_LLM_Model()


def load_vocabulary(model: Small_LLM_Model) -> Vocabulary:
    """Load token text via the public SDK path helpers.

    Args:
        model: Initialized ``Small_LLM_Model``.

    Returns:
        A ``Vocabulary`` used to test tokens against the schema.

    Raises:
        RuntimeError: If neither vocab file can be read.
    """
    errors: list[str] = []
    getters = (
        model.get_path_to_vocab_file,
        model.get_path_to_tokenizer_file,
    )
    for getter in getters:
        try:
            return Vocabulary.from_file(getter())
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(str(exc))
    joined = "; ".join(errors) if errors else "unknown error"
    raise RuntimeError(f"could not load tokenizer vocabulary ({joined})")


def legal_token_ids(
    state: DecodeState,
    vocab: Vocabulary,
    vocab_size: int,
) -> list[int]:
    """Return token ids that keep *state* on a valid JSON path.

    Args:
        state: Current constraint machine.
        vocab: Id-to-text table.
        vocab_size: Length of the logits vector.

    Returns:
        Legal token ids, possibly empty.
    """
    legal: list[int] = []
    for token_id in vocab.candidate_ids(state.allowed_first_chars()):
        if token_id >= vocab_size:
            continue
        if state.accepts(vocab.text_of(token_id)):
            legal.append(token_id)
    return legal


def pick_token(logits: list[float], legal: list[int]) -> int | None:
    """Argmax over *legal* ids after setting the rest to -inf.

    Args:
        logits: Raw next-token scores from the model.
        legal: Token ids allowed by the constraint machine.

    Returns:
        The winning token id, or ``None`` if nothing legal remains.
    """
    if not legal:
        return None
    scores = np.asarray(logits, dtype=np.float64)
    size = int(scores.shape[0])
    masked = np.full(size, -np.inf, dtype=np.float64)
    valid = np.array(
        [token_id for token_id in legal if 0 <= token_id < size],
        dtype=np.int64,
    )
    if valid.size == 0:
        return None
    masked[valid] = scores[valid]
    chosen = int(np.argmax(masked))
    if not np.isfinite(masked[chosen]):
        return None
    return chosen


def _emergency_finish(state: DecodeState) -> None:
    """Force remaining template characters so the JSON always parses."""
    for _ in range(_EMERGENCY_CHARS):
        if state.is_complete():
            return
        firsts = state.allowed_first_chars()
        if firsts is None:
            char = '"'
        elif not firsts:
            return
        else:
            char = sorted(firsts)[0]
        if not state.feed_char(char, record=True):
            return


def generate_call(
    model: Small_LLM_Model,
    vocab: Vocabulary,
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> OutputRecord:
    """Generate one schema-valid function call for *user_prompt*.

    Args:
        model: LLM wrapper (logits + encode only).
        vocab: Tokenizer vocabulary.
        functions: Allowed tools.
        user_prompt: Natural-language request.

    Returns:
        An ``OutputRecord`` ready to write to disk.

    Raises:
        RuntimeError: If the decoder cannot produce parseable JSON.
        ValueError: If the finished text is not a JSON object.
    """
    prompt = build_generation_prompt(functions, user_prompt)
    ids = tensor_to_ids(model.encode(prompt))
    state = DecodeState.start(functions)
    new_tokens = 0
    while not state.is_complete() and new_tokens < _MAX_NEW_TOKENS:
        logits = model.get_logits_from_input_ids(ids)
        legal = legal_token_ids(state, vocab, len(logits))
        token_id = pick_token(logits, legal)
        if token_id is None:
            break
        state.advance(vocab.text_of(token_id))
        ids.append(token_id)
        new_tokens += 1
    if not state.is_complete():
        _emergency_finish(state)
    if not state.is_complete():
        raise RuntimeError("JSON object was not closed")
    parsed: object = json.loads(state.generated)
    if not isinstance(parsed, dict):
        raise ValueError("decoder produced a non-object JSON value")
    return OutputRecord.model_validate(
        {
            "prompt": user_prompt,
            "name": parsed.get("name"),
            "parameters": parsed.get("parameters"),
        }
    )


def run_call_decoding(
    functions: list[FunctionDefinition],
    prompts: list[TestPrompt],
) -> list[OutputRecord]:
    """Load the model once, then decode a full function call per prompt.

    Prints each record. The caller writes ``--output``.

    Args:
        functions: Catalog from ``--functions_definition``.
        prompts: Requests from ``--input``.

    Returns:
        One output object per prompt, in the same order.
    """
    try:
        model = load_model()
        vocab = load_vocabulary(model)
        print(f"loaded {len(vocab.all_ids)} vocab entries")
        records: list[OutputRecord] = []
        total = len(prompts)
        for index, item in enumerate(prompts, start=1):
            print(f"decoding {index}/{total}...", flush=True)
            record = generate_call(model, vocab, functions, item.prompt)
            records.append(record)
            print(f"  {record.model_dump()}")
        return records
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: constrained decoding failed ({exc})", file=sys.stderr)
        raise SystemExit(1) from exc
