"""Function calling / tool use."""

import json

from src.client import MODEL_ID, get_client

client = get_client()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo, Japan'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

messages = [{"role": "user", "content": "What is the weather like in Tokyo right now?"}]

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages,
    tools=tools,
    max_tokens=8192,
)

assistant_msg = response.choices[0].message

if assistant_msg.tool_calls:
    tool_call = assistant_msg.tool_calls[0]
    print(f"Tool called: {tool_call.function.name}")
    print(f"Arguments:   {tool_call.function.arguments}")

    # Simulate the tool response
    tool_result = json.dumps({"temperature": 22, "condition": "Partly cloudy", "unit": "celsius"})

    messages = [
        *messages,
        assistant_msg,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result,
        },
    ]

    followup = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        max_tokens=8192,
    )

    print()
    print("=== Final answer ===")
    print(followup.choices[0].message.content)
else:
    print(assistant_msg.content)
