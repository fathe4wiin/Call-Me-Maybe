"""Entry point for `uv run python -m src` / `python -m src`."""

from .cli import CliArgs
from .load import JsonLoader


def main() -> None:
    """Parse flags, load both JSON files, print a short sanity check."""
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
    print(f"output will be written later to {args.output}")


if __name__ == "__main__":
    main()
