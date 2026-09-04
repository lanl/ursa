from pydantic import BaseModel, Field

from ursa.util.structured_output import (
    StructuredOutputResult,
    invoke_structured,
)


class ExampleSchema(BaseModel):
    ok: bool = Field(description="Whether the example succeeded.")
    reason: str = Field(description="Reason.")


class FakeStructuredLLM:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, input, **kwargs):
        self.parent.invocations.append((input, kwargs))
        response = self.parent.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.invocations = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append((schema, kwargs))
        return FakeStructuredLLM(self)


def test_invoke_structured_retries_function_calling_after_none():
    llm = FakeLLM([
        None,
        {"ok": True, "reason": "function calling worked"},
    ])

    result = invoke_structured(
        llm,
        ExampleSchema,
        "prompt",
        context="test schema",
        repair=False,
    )

    assert isinstance(result, ExampleSchema)
    assert result.ok is True
    assert result.reason == "function calling worked"
    assert llm.calls[0] == (ExampleSchema, {})
    assert llm.calls[1] == (ExampleSchema, {"method": "function_calling"})


def test_invoke_structured_returns_validated_fallback_after_failures():
    llm = FakeLLM([None, None])

    result = invoke_structured(
        llm,
        ExampleSchema,
        "prompt",
        context="fallback test",
        fallback={"ok": False, "reason": "fallback used"},
        repair=False,
    )

    assert isinstance(result, ExampleSchema)
    assert result.ok is False
    assert result.reason == "fallback used"


def test_invoke_structured_repairs_include_raw_parsed_none():
    llm = FakeLLM([
        {"raw": "not valid", "parsed": None, "parsing_error": "bad json"},
        {
            "raw": "valid after repair",
            "parsed": {"ok": True, "reason": "repaired"},
            "parsing_error": None,
        },
    ])

    result = invoke_structured(
        llm,
        ExampleSchema,
        "original prompt",
        include_raw=True,
        methods=(None,),
        context="repair test",
        repair=True,
    )

    assert isinstance(result, StructuredOutputResult)
    assert isinstance(result.parsed, ExampleSchema)
    assert result.parsed.ok is True
    assert result.raw == "valid after repair"
    assert len(llm.calls) == 2
    repair_messages = llm.invocations[1][0]
    assert "Repair attempt: 1 of 1" in repair_messages[1].content
    assert "bad json" in repair_messages[1].content


def test_invoke_structured_accepts_integer_repair_attempts():
    llm = FakeLLM([
        None,
        None,
        {"ok": True, "reason": "second repair worked"},
    ])

    result = invoke_structured(
        llm,
        ExampleSchema,
        "prompt",
        methods=(None,),
        context="repair count test",
        repair=2,
    )

    assert isinstance(result, ExampleSchema)
    assert result.reason == "second repair worked"
    assert len(llm.calls) == 3
