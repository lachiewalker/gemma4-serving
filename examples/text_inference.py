"""Basic text inference."""

from src.client import MODEL_ID, get_client

client = get_client()

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {"role": "user", "content": "Explain the three laws of thermodynamics in plain English."},
    ],
    max_tokens=8192,
    temperature=0.7,
)

print(response.choices[0].message.content)
