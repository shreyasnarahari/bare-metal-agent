from copy import deepcopy

from toolsmith.prompting import ChatPromptRenderer


class FakeTokenizer:
    def __init__(self):
        self.messages = None
        self.options = None

    def apply_chat_template(self, messages, **options):
        self.messages = deepcopy(messages)
        self.options = {**options, "tools": deepcopy(options["tools"])}
        messages[0]["content"] = "changed by tokenizer"
        if options["tools"]:
            options["tools"][0]["function"]["name"] = "changed"
        return "rendered prompt"


def test_renderer_uses_native_template_without_mutating_inputs():
    tokenizer = FakeTokenizer()
    renderer = ChatPromptRenderer(tokenizer)
    messages = [{"role": "user", "content": "Weather?"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]

    prompt = renderer.render(messages, tools, enable_thinking=False)

    assert prompt == "rendered prompt"
    assert messages == [{"role": "user", "content": "Weather?"}]
    assert tools[0]["function"]["name"] == "get_weather"
    assert tokenizer.options == {
        "tools": tools,
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_renderer_preserves_assistant_calls_and_tool_results():
    tokenizer = FakeTokenizer()
    renderer = ChatPromptRenderer(tokenizer)
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature":22}'},
    ]

    renderer.render(messages, enable_thinking=True)

    assert tokenizer.messages == messages
    assert tokenizer.options["tools"] is None
    assert tokenizer.options["enable_thinking"] is True
