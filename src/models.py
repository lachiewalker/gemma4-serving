from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    TITLE = "title"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    PARAGRAPH = "paragraph"
    CAPTION = "caption"
    HEADER = "header"       # recurring page header band
    FOOTER = "footer"       # recurring page footer band
    PAGE_NUMBER = "page_number"
    FIGURE = "figure"       # visual description of a figure / illustration
    TABLE = "table"         # table content rendered as markdown
    LIST_ITEM = "list_item"
    CODE = "code"
    FOOTNOTE = "footnote"


class DocumentBlock(BaseModel):
    type: BlockType
    text: str = Field(description="Extracted text content of this block")


class PageOCR(BaseModel):
    """Blocks extracted from a single document page, in reading order."""

    blocks: List[DocumentBlock] = Field(
        description=(
            "Every text element on this page in reading order "
            "(top-to-bottom, left-to-right). "
            "Produce one block per distinct element — do not merge separate paragraphs. "
            "Use type='figure' for image/chart descriptions and "
            "type='table' for tables formatted as a markdown table."
        )
    )


class DocumentOCR(BaseModel):
    """Aggregated OCR result for a complete document."""

    title: Optional[str] = Field(None, description="Main document title, if identifiable")
    language: Optional[str] = Field(None, description="ISO 639-1 language code, e.g. 'en'")
    page_count: int = Field(description="Total number of pages in the document")
    pages: List[PageOCR] = Field(description="Per-page extraction results, ordered by page number")
