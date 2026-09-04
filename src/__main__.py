"""Entry point for `uv run python -m src` / `python -m src`."""

from __future__ import annotations

import sys

from .cli import CliArgs
from .generate import run_call_decoding
from .load import JsonLoader
from .save import save_results


def main() -> None:
    """Parse flags, load JSON, decode function calls, write the output file."""
    args = CliArgs.parse()
    catalog = JsonLoader(path=args.functions_definition).load_functions()
    prompts = JsonLoader(path=args.input).load_prompts()

    print(f"loaded {len(catalog)} function(s) from {args.functions_definition}")
    for function in catalog:
        param_names = ", ".join(
            f"{name}: {schema.type}" for name, schema in function.parameters.items()
        )
        print(f"  - {function.name}({param_names}) -> {function.returns.type}")

    print(f"loaded {len(prompts)} prompt(s) from {args.input}")

    if not catalog:
        print("error: function catalog is empty", file=sys.stderr)
        raise SystemExit(1)

    records = run_call_decoding(catalog, prompts)
    try:
        save_results(args.output, records)
    except OSError as exc:
        reason = exc.strerror or str(exc)
        print(
            f"error: {args.output}: cannot write file ({reason})",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    print(f"wrote {len(records)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
