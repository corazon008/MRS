from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_document(path: Path) -> list[Document]:
    """Load a PDF into LangChain Documents."""
    loader = None
    match path.suffix.lower():
        case ".pdf":
            loader = PyPDFLoader(path)
        case ".txt":
            loader = TextLoader(path)
        case _:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader.load()


def load_documents(directory: str | Path) -> list[Document]:
    """Load all PDFs from a directory."""
    directory = Path(directory)

    documents = []

    for path in directory.glob("*.pdf"):
        documents.extend(load_document(path))

    for path in directory.glob("*.txt"):
        documents.extend(load_document(path))

    return documents
