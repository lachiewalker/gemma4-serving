"""Document OCR via vLLM structured outputs.

One API call per page — keeps the model focused on a bounded amount of content,
avoids token budget exhaustion on long documents, and enables parallel processing.

Supported inputs:
  - PDF  (converted to page images via pymupdf)
  - PNG / JPEG / WEBP  (treated as a single page)
"""

import base64
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.client import MODEL_ID, get_client
from src.models import BlockType, DocumentOCR, PageOCR

# Server is configured with max_model_len=32768.
# Each page call gets a generous budget — a dense A4 page of text typically
# produces 1000-3000 tokens of structured output; 16384 gives ample headroom
# for tables and figures without approaching the server limit.
_MAX_TOKENS_PER_PAGE = 16384

_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}

_SYSTEM_PROMPT = (
    "You are a precise document OCR engine. "
    "Extract every text element from the provided document page image in reading order "
    "(top-to-bottom, left-to-right). "
    "Produce one block per distinct element — do not merge separate paragraphs into one block. "
    "For figures and illustrations, write a concise visual description as the text. "
    "For tables, render the full content as a markdown table. "
    "Do not skip any text visible in the image."
)


def _image_to_data_url(image_bytes: bytes, suffix: str) -> str:
    mime = _MIME.get(suffix.lstrip(".").lower(), "image/jpeg")
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _pdf_to_page_images(pdf_path: Path) -> list[bytes]:
    """Render each PDF page to a PNG at 150 dpi."""
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError("Install pymupdf to process PDFs: uv pip install pymupdf") from e

    doc = fitz.open(str(pdf_path))
    pages = []
    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 dpi
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(pix.tobytes("png"))
    return pages


def ocr_page(image_bytes: bytes, suffix: str = "png") -> PageOCR:
    """Run OCR on a single page image. Returns structured blocks."""
    data_url = _image_to_data_url(image_bytes, suffix)
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract all content from this page."},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "page-ocr",
                "schema": PageOCR.model_json_schema(),
            },
        },
        max_tokens=_MAX_TOKENS_PER_PAGE,
        temperature=0.0,
    )

    return PageOCR.model_validate(json.loads(response.choices[0].message.content))


def ocr_document(
    path: str | Path,
    max_workers: int = 4,
) -> DocumentOCR:
    """Extract structured content from a PDF or image file.

    Pages are processed in parallel (up to max_workers concurrent API calls).
    """
    path = Path(path)
    suffix = path.suffix.lstrip(".").lower()

    if suffix == "pdf":
        page_images = _pdf_to_page_images(path)
        img_suffix = "png"
    else:
        page_images = [path.read_bytes()]
        img_suffix = suffix

    page_count = len(page_images)
    results: dict[int, PageOCR] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(ocr_page, img, img_suffix): idx
            for idx, img in enumerate(page_images)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    pages = [results[i] for i in range(page_count)]

    # Infer document title from the first TITLE block across all pages
    title: Optional[str] = None
    for page in pages:
        for block in page.blocks:
            if block.type == BlockType.TITLE:
                title = block.text
                break
        if title:
            break

    return DocumentOCR(
        title=title,
        language=None,  # could add a cheap follow-up call to detect language
        page_count=page_count,
        pages=pages,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr <pdf_or_image_path>")
        sys.exit(1)

    result = ocr_document(sys.argv[1])
    print(f"Title:  {result.title}")
    print(f"Pages:  {result.page_count}")
    print()
    for i, page in enumerate(result.pages, 1):
        print(f"--- Page {i} ({len(page.blocks)} blocks) ---")
        for block in page.blocks:
            print(f"  [{block.type.value}] {block.text[:100]}")
    print()
    print("Full JSON:")
    print(result.model_dump_json(indent=2))
