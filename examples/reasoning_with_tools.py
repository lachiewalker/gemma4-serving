"""Combined reasoning + tool use — thinking mode with function calling."""

import json

from src.client import MODEL_ID, get_client

client = get_client()

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the numeric result",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate, e.g. '2 ** 10 + sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": (
            "A data centre rack holds 42 units. "
            "Each 1U server draws 300 W at full load. "
            "If the rack is 80% full, what is the total power draw in kilowatts? "
            "Then calculate how many such racks you could run on a 1 MW power budget."
        ),
    }
]

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages,
    tools=tools,
    max_tokens=8192,
)

msg = response.choices[0].message

if hasattr(msg, "reasoning") and msg.reasoning:
    print("=== Thinking ===")
    print(msg.reasoning)
    print()

if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"Tool: {tc.function.name}({tc.function.arguments})")
        args = json.loads(tc.function.arguments)
        # Simple eval — safe enough for demo numbers
        result = str(eval(args["expression"], {"__builtins__": {}}, {"sqrt": __import__("math").sqrt}))  # noqa: S307
        print(f"Result: {result}")

        messages = [
            *messages,
            msg,
            {"role": "tool", "tool_call_id": tc.id, "content": result},
        ]

    followup = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        tools=tools,
        max_tokens=8192,
    
    )

    final = followup.choices[0].message
    if hasattr(final, "reasoning") and final.reasoning:
        print()
        print("=== Follow-up thinking ===")
        print(final.reasoning)

    print()
    print("=== Answer ===")
    print(final.content)
else:
    print("=== Answer ===")
    print(msg.content)
