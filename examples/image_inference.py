"""Image understanding — pass a URL or local file path as an argument."""

import base64
import sys
from pathlib import Path

from src.client import MODEL_ID, get_client

client = get_client()

if len(sys.argv) > 1 and not sys.argv[1].startswith("http"):
    # Local file → base64
    data = Path(sys.argv[1]).read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = Path(sys.argv[1]).suffix.lstrip(".").lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
    image_content = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
else:
    url = sys.argv[1] if len(sys.argv) > 1 else "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
    image_content = {"type": "image_url", "image_url": {"url": url}}

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ],
    max_tokens=1024,
    temperature=0.0,
)

print(response.choices[0].message.content)
