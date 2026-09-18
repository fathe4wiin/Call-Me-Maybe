"""Steering text: catalog + user request. Not a chat system role."""

from __future__ import annotations

from .models import FunctionDefinition, ParamSchema

JSON_PREFIX = '{"name":"'

UNKNOWN_FUNCTION_NAME = "unknown"

UNKNOWN_FUNCTION = FunctionDefinition(
    name=UNKNOWN_FUNCTION_NAME,
    description=(
        "Fallback when none of the other listed functions can fulfill "
        "the request. Choose this only if no other function matches."
    ),
    parameters={},
    returns=ParamSchema(type="string"),
)


def with_unknown_function(
    functions: list[FunctionDefinition],
) -> list[FunctionDefinition]:
    """Return *functions* plus the unknown fallback if that name is free.

    Args:
        functions: Tools loaded from ``functions_definition.json``.

    Returns:
        The catalog, with ``unknown`` appended unless it is already present.
    """
    if any(item.name == UNKNOWN_FUNCTION_NAME for item in functions):
        return functions
    return [*functions, UNKNOWN_FUNCTION]


def build_steering_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build the plain-text instructions the model sees before JSON.

    The SDK has no chat template. This string is the whole prompt: a short
    instruction, every catalog function, then the user request.

    Args:
        functions: Tools loaded from ``functions_definition.json``.
        user_prompt: Natural-language request from the tests file.

    Returns:
        Completion text ending just before the JSON object.
    """
    lines: list[str] = [
        "Translate the user request into a JSON function call.",
        "Use exactly one function from the list below.",
        "If a listed function matches the request, use that function.",
        "If none of the listed functions match, prefer unknown.",
        "",
        "Available functions:",
    ]
    for function in with_unknown_function(functions):
        params = ", ".join(
            f"{name}: {schema.type}"
            for name, schema in function.parameters.items()
        )
        if not params:
            params = "(none)"
        lines.append(f"- {function.name}: {function.description}")
        lines.append(f"  parameters: {params}")
        lines.append(f"  returns: {function.returns.type}")
    lines.extend(
        [
            "",
            "User request:",
            user_prompt,
            "",
            "JSON function call:",
        ]
    )
    return "\n".join(lines)


def build_generation_prompt(
    functions: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Steering text plus the forced JSON prefix ``{"name":"``.

    Args:
        functions: Tools loaded from ``functions_definition.json``.
        user_prompt: Natural-language request from the tests file.

    Returns:
        Text that ``encode`` will turn into input ids.
    """
    return build_steering_prompt(functions, user_prompt) + "\n" + JSON_PREFIX
