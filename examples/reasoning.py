"""Reasoning / thinking mode — the model reasons step-by-step before answering."""

from src.client import MODEL_ID, get_client

client = get_client()

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": (
                "A snail is at the bottom of a 20-foot well. "
                "Each day it climbs 3 feet, but each night it slides back 2 feet. "
                "How many days will it take to reach the top?"
            ),
        }
    ],
    max_tokens=8192,
)

message = response.choices[0].message

if hasattr(message, "reasoning") and message.reasoning:
    print("=== Thinking ===")
    print(message.reasoning)
    print()

print("=== Answer ===")
print(message.content)
