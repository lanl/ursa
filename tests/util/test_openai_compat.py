"""Responses API -> Chat Completions fallback (Nemotron-via-LiteLLM case).

Drives the real ChatOpenAI/openai SDK stack against ``httpx.MockTransport`` so
routing, retries and the exact request bodies are exercised without network.
"""

import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from ursa.util import openai_compat
from ursa.util.openai_compat import (
    ResponsesFallbackChatOpenAI,
    _relocate_tool_images,
    with_responses_fallback,
)

IMG = {"type": "image", "base64": "iVBORw0KGgo", "mime_type": "image/png"}
# Body shape of the real failure: LiteLLM -> vLLM rejecting a Responses
# ``input_image`` inside a ``function_call_output``.
NEMOTRON_400 = {
    "error": {
        "message": (
            "litellm.BadRequestError: 1 validation error: [type=missing, "
            "input_value={'type': 'input_image', 'image_url': 'data:...'}, "
            "input_type=dict] LiteLLM: model group 'Nemotron' failed"
        ),
        "type": "invalid_request_error",
        "param": None,
        "code": "400",
    }
}
CHAT_OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "nemotron",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def _tool_image_history():
    return [
        HumanMessage("plot it"),
        AIMessage(
            "", tool_calls=[{"name": "read_image", "args": {}, "id": "call_1"}]
        ),
        ToolMessage(content=["see plot", IMG], tool_call_id="call_1"),
    ]


@pytest.fixture(autouse=True)
def _clear_pins():
    openai_compat._CHAT_COMPLETIONS_PINNED.clear()
    yield
    openai_compat._CHAT_COMPLETIONS_PINNED.clear()


class _Server:
    """Records requests; answers /responses and /chat/completions."""

    def __init__(self, responses=(400, NEMOTRON_400), chat=(200, CHAT_OK)):
        self.requests: list[tuple[str, dict]] = []
        self.responses, self.chat = responses, chat

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        path = request.url.path
        self.requests.append((path, body))
        status, payload = (
            self.responses if path.endswith("/responses") else self.chat
        )
        if body.get("stream") and status == 200:
            chunk = {
                **CHAT_OK,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "done"},
                        "finish_reason": None,
                    }
                ],
            }
            sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
            return httpx.Response(
                200, text=sse, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(status, json=payload)

    @property
    def paths(self):
        return [p.rsplit("/", 1)[-1] for p, _ in self.requests]


def _llm(server, use_responses_api=True, **kwargs):
    transport = httpx.MockTransport(server)
    base = ChatOpenAI(
        model="nemotron",
        base_url="http://litellm.test/v1",
        api_key="sk-test",
        use_responses_api=use_responses_api,
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
        **kwargs,
    )
    return with_responses_fallback(base)


def test_upgrade_is_in_place_and_preserves_configuration():
    server = _Server()
    transport = httpx.MockTransport(server)
    base = ChatOpenAI(
        model="nemotron",
        base_url="http://litellm.test/v1",
        api_key="sk-test",
        use_responses_api=True,
        temperature=0.2,
        max_retries=0,
        http_client=httpx.Client(transport=transport),
    )
    base._ursa_context_source = (
        "Configured profile"  # set by load_model_context
    )
    client = base.root_client
    llm = with_responses_fallback(base)
    assert llm is base  # same object: references and attributes stay valid
    assert isinstance(llm, ResponsesFallbackChatOpenAI)
    assert isinstance(llm, ChatOpenAI)
    assert llm.root_client is client
    assert llm._ursa_context_source == "Configured profile"
    assert llm.model_name == "nemotron"
    assert llm.openai_api_base == "http://litellm.test/v1"
    assert llm.use_responses_api is True and llm.temperature == 0.2
    assert llm.openai_api_key.get_secret_value() == "sk-test"
    llm.invoke("hi")  # custom http_client (mock transport) is still in use
    assert server.requests


def test_non_chatopenai_models_are_returned_unchanged():
    sentinel = object()
    assert with_responses_fallback(sentinel) is sentinel
    already = with_responses_fallback(_llm(_Server()))
    assert with_responses_fallback(already) is already


def test_falls_back_and_relocates_tool_images():
    server = _Server()
    llm = _llm(server)

    reply = llm.invoke(_tool_image_history())

    assert reply.text == "done"
    assert server.paths == ["responses", "completions"]
    chat_messages = server.requests[1][1]["messages"]
    tool = next(m for m in chat_messages if m["role"] == "tool")
    assert isinstance(tool["content"], str)  # text only, as the spec requires
    assert "see plot" in tool["content"]
    follow = chat_messages[chat_messages.index(tool) + 1]
    assert follow["role"] == "user"
    assert any(part.get("type") == "image_url" for part in follow["content"])


def test_pin_skips_responses_for_later_calls_and_copies():
    server = _Server()
    llm = _llm(server)
    llm.invoke("first")
    llm.model_copy().bind_tools([]).invoke("second")  # how agents call it
    assert server.paths == ["responses", "completions", "completions"]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, {"error": {"message": "invalid api key", "type": "auth"}}),
        (429, {"error": {"message": "rate limited", "type": "rate"}}),
        (500, {"error": {"message": "upstream exploded", "type": "server"}}),
        (400, {"error": {"message": "Invalid tool schema", "param": "tools"}}),
        (
            400,
            {
                "error": {
                    "message": "maximum context length is 8192 tokens",
                    "code": "context_length_exceeded",
                }
            },
        ),
    ],
)
def test_unrelated_errors_never_fall_back(status, error):
    server = _Server(responses=(status, error))
    with pytest.raises(Exception):  # noqa: B017 - exact class varies by status
        _llm(server).invoke("hi")
    assert server.paths == ["responses"]
    assert not openai_compat._CHAT_COMPLETIONS_PINNED


def test_failed_fallback_raises_original_and_does_not_pin():
    server = _Server(chat=(400, {"error": {"message": "chat also broken"}}))
    with pytest.raises(Exception) as info:
        _llm(server).invoke("hi")
    assert "input_image" in str(info.value)  # original Responses error
    assert server.paths == ["responses", "completions"]
    assert not openai_compat._CHAT_COMPLETIONS_PINNED


def test_missing_responses_route_falls_back():
    server = _Server(responses=(404, {"error": {"message": "Not Found"}}))
    assert _llm(server).invoke("hi").text == "done"
    assert server.paths == ["responses", "completions"]


async def test_async_falls_back():
    server = _Server()
    reply = await _llm(server).ainvoke(_tool_image_history())
    assert reply.text == "done"
    assert server.paths == ["responses", "completions"]


def test_stream_falls_back_before_first_chunk():
    server = _Server()
    text = "".join(chunk.text for chunk in _llm(server).stream("hi"))
    assert text == "done"
    assert server.paths == ["responses", "completions"]


def test_responses_path_keeps_images_in_tool_output():
    # Real OpenAI Responses supports images in function_call_output: untouched.
    # Image-only content is what URSA's read_image tool returns.
    history = _tool_image_history()
    history[-1] = ToolMessage(content=[IMG], tool_call_id="call_1")
    payload = _llm(_Server())._get_request_payload(history)
    output = next(
        i for i in payload["input"] if i["type"] == "function_call_output"
    )
    assert any(
        isinstance(part, dict) and part.get("type") == "input_image"
        for part in output["output"]
    )


def test_relocation_inserts_one_user_message_after_the_tool_run():
    history = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "a", "args": {}, "id": "c1"},
                {"name": "b", "args": {}, "id": "c2"},
            ],
        ),
        ToolMessage(content=[IMG], tool_call_id="c1"),
        ToolMessage(content=["text only"], tool_call_id="c2"),
        HumanMessage("next"),
    ]
    out = _relocate_tool_images(history)
    assert [type(m).__name__ for m in out] == [
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "HumanMessage",
        "HumanMessage",
    ]
    assert out[1].content == "[image output attached in the next message]"
    assert out[2] is history[2]  # no image -> untouched
    assert out[3].content[1] == IMG


def test_subclass_adds_no_state_so_in_place_upgrade_stays_safe():
    # with_responses_fallback swaps __class__; that is only sound while the
    # subclass declares no fields/private attributes of its own.
    assert set(ResponsesFallbackChatOpenAI.model_fields) == set(
        ChatOpenAI.model_fields
    )
    assert set(ResponsesFallbackChatOpenAI.__private_attributes__) == set(
        ChatOpenAI.__private_attributes__
    )


REASONING = {"effort": "medium", "summary": "auto"}


def test_reasoning_is_mapped_for_chat_completions():
    # Regression: ``reasoning`` in the config reached Completions.create() and
    # raised TypeError before any request was sent.
    server = _Server()
    reply = _llm(server, use_responses_api=False, reasoning=REASONING).invoke(
        "hi"
    )
    assert reply.text == "done" and server.paths == ["completions"]
    body = server.requests[0][1]
    assert body["reasoning_effort"] == "medium"
    assert "reasoning" not in body


def test_fallback_succeeds_with_reasoning_configured():
    # The Nemotron case: Responses 400s, and the retry must not trip over the
    # Responses-only ``reasoning`` setting.
    server = _Server()
    reply = _llm(server, reasoning=REASONING).invoke(_tool_image_history())
    assert reply.text == "done"
    assert server.paths == ["responses", "completions"]
    assert server.requests[0][1]["reasoning"] == REASONING  # untouched there
    retry = server.requests[1][1]
    assert retry["reasoning_effort"] == "medium" and "reasoning" not in retry


def test_explicit_reasoning_effort_wins_and_other_params_are_dropped():
    server = _Server()
    llm = _llm(
        server,
        use_responses_api=False,
        reasoning=REASONING,
        reasoning_effort="low",
        include=["reasoning.encrypted_content"],
        truncation="auto",
    )
    llm.invoke("hi")
    body = server.requests[0][1]
    assert body["reasoning_effort"] == "low"
    for key in ("reasoning", "include", "truncation"):
        assert key not in body


def test_no_reasoning_configured_adds_nothing():
    server = _Server()
    _llm(server, use_responses_api=False).invoke("hi")
    assert "reasoning_effort" not in server.requests[0][1]
