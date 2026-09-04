"""Steering text: catalog + user request. Not a chat system role."""

from __future__ import annotations

from .models import FunctionDefinition

JSON_PREFIX = '{"name":"'


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
        "",
        "Available functions:",
    ]
    for function in functions:
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
