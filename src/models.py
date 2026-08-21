"""Pydantic models for the function catalog, test prompts, and output records."""

from pydantic import BaseModel, Field


class ParamSchema(BaseModel):
    """JSON object like ``{"type": "number"}`` or ``{"type": "string"}``."""

    type: str = Field(description="JSON type name, e.g. number, string, boolean.")


class FunctionDefinition(BaseModel):
    """One tool from ``functions_definition.json``."""

    name: str
    description: str
    parameters: dict[str, ParamSchema]
    returns: ParamSchema


class TestPrompt(BaseModel):
    """One object from ``function_calling_tests.json``."""

    prompt: str


class OutputRecord(BaseModel):
    """One object written to the output JSON after decoding."""

    prompt: str
    name: str
    parameters: dict[str, object]
