"""Document OCR — pass a PDF or image path as an argument.

Pages are processed in parallel. Each page gets its own API call with a
generous token budget, so dense pages are fully transcribed rather than
summarised.
"""

import sys

from src.ocr import ocr_document

if len(sys.argv) < 2:
    print("Usage: python examples/document_ocr.py <pdf_or_image_path> [max_workers]")
    sys.exit(1)

max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
result = ocr_document(sys.argv[1], max_workers=max_workers)

print(f"Title:    {result.title}")
print(f"Language: {result.language}")
print(f"Pages:    {result.page_count}")
print()

for i, page in enumerate(result.pages, 1):
    print(f"--- Page {i} ({len(page.blocks)} blocks) ---")
    for block in page.blocks:
        print(f"  [{block.type.value}] {block.text[:120]}")
