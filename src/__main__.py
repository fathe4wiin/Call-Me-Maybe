"""Entry point for `uv run python -m src` / `python -m src`."""

from __future__ import annotations

import signal
import sys

from .cli import CliArgs
from .generate import run_call_decoding
from .load import JsonLoader
from .save import save_results

_QUIT_SIGNAL_NAMES = ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT")


def _handle_quit(signum: int, _frame: object) -> None:
    """Print one line and exit. Used for Ctrl+C and other quit signals."""
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    print(f"error: interrupted ({name})", file=sys.stderr)
    raise SystemExit(128 + signum)


def _install_quit_handlers() -> None:
    """Catch process-quit signals so they do not dump a traceback."""
    for name in _QUIT_SIGNAL_NAMES:
        sig = getattr(signal, name, None)
        if not isinstance(sig, int):
            continue
        try:
            signal.signal(sig, _handle_quit)
        except (OSError, ValueError):
            continue


def main() -> None:
    """Parse flags, load JSON, decode function calls, write the output file."""
    _install_quit_handlers()
    args = CliArgs.parse()
    catalog = JsonLoader(path=args.functions_definition).load_functions()
    prompts = JsonLoader(path=args.input).load_prompts()

    print(
        f"loaded {len(catalog)} function(s) from "
        f"{args.functions_definition}"
    )
    for function in catalog:
        param_names = ", ".join(
            f"{name}: {schema.type}"
            for name, schema in function.parameters.items()
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
    _install_quit_handlers()
    try:
        main()
    except KeyboardInterrupt:
        print("error: interrupted (SIGINT)", file=sys.stderr)
        raise SystemExit(130) from None
