"""Loaders that turn files on disk into :class:`~rag_kb.schemas.Document` objects.

Supported formats: ``.txt``, ``.md``, and (optionally) ``.pdf`` when the
``pypdf`` package is installed. Markdown files are split into one document
per heading so section metadata survives chunking.
"""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from rag_kb.schemas import Document

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = frozenset({".txt", ".md", ".pdf"})
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def load_path(path: str | Path) -> list[Document]:
    """Load a file, or every supported file under a directory.

    Args:
        path: A file or directory. Directories are walked recursively and
            files are processed in sorted order for determinism.

    Returns:
        The loaded documents. Markdown files may yield several documents
        (one per section); PDFs yield one document per page.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is a file with an unsupported extension.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such file or directory: {root}")
    if root.is_file():
        return list(_load_file(root))

    documents: list[Document] = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file.suffix.lower() in _SUPPORTED_SUFFIXES:
            documents.extend(_load_file(file))
    logger.info("Loaded %d document(s) from %s", len(documents), root)
    return documents


def _load_file(path: Path) -> Iterator[Document]:
    """Dispatch a single file to the loader for its extension."""
    suffix = path.suffix.lower()
    if suffix == ".md":
        yield from _load_markdown(path)
    elif suffix == ".txt":
        yield from _load_text(path)
    elif suffix == ".pdf":
        yield from _load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type '{suffix}': {path}")


def _load_text(path: Path) -> Iterator[Document]:
    """Load a plain-text file as a single document."""
    text = path.read_text(encoding="utf-8").strip()
    if text:
        yield Document(text=text, source=path.name)


def _load_markdown(path: Path) -> Iterator[Document]:
    """Load a Markdown file, yielding one document per heading section.

    Text before the first heading is yielded with ``section=None``.
    """
    section: str | None = None
    buffer: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADING_RE.match(line)
        if match:
            yield from _flush_section(buffer, path.name, section)
            section = match.group(2)
            buffer = []
        else:
            buffer.append(line)
    yield from _flush_section(buffer, path.name, section)


def _flush_section(
    lines: list[str], source: str, section: str | None
) -> Iterator[Document]:
    """Yield a document for the accumulated section lines, if non-empty."""
    text = "\n".join(lines).strip()
    if text:
        yield Document(text=text, source=source, section=section)


def _load_pdf(path: Path) -> Iterator[Document]:
    """Load a PDF, yielding one document per non-empty page.

    Raises:
        ImportError: If the optional ``pypdf`` dependency is missing.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PDF support requires the optional 'pypdf' package; "
            "install it with `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(path))
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            yield Document(text=text, source=path.name, page=number)
