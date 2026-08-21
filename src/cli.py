"""CLI: --functions_definition, --input, --output and their defaults."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel

DEFAULT_FUNCTIONS_DEFINITION = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


class CliArgs(BaseModel):
    """Paths taken from the command line, with subject defaults."""

    functions_definition: Path
    input: Path
    output: Path

    @classmethod
    def parse(cls) -> CliArgs:
        """Parse argv into three paths. Does not open any files."""
        parser = argparse.ArgumentParser(
            description="Translate natural-language prompts into function calls.",
        )
        parser.add_argument(
            "--functions_definition",
            type=Path,
            default=DEFAULT_FUNCTIONS_DEFINITION,
            help="JSON array of available functions (default: %(default)s)",
        )
        parser.add_argument(
            "--input",
            type=Path,
            default=DEFAULT_INPUT,
            help="JSON array of prompts (default: %(default)s)",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=DEFAULT_OUTPUT,
            help="Where to write results later (default: %(default)s)",
        )
        args = parser.parse_args()
        return cls(
            functions_definition=args.functions_definition,
            input=args.input,
            output=args.output,
        )
