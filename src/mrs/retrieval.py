from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
import os
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaEmbeddings

from mrs.constants import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)

RERANKER_MODEL = os.environ.get("MAIA_RERANKER_MODEL", "BAAI/bge-reranker-base")

model = CrossEncoder(RERANKER_MODEL, max_length=512)

SMALL_SET_SKIP = (
    3  # Skip reranking if the candidate pool is smaller than this threshold
)


class Retriever:
    """Retriever class to handle vector store and retrieval operations."""

    def __init__(
        self, embeddings: Embeddings | str, recreate_collection: bool = False
    ):
        """
        Initialize the Retriever with embeddings.
        @param embeddings: An instance of Embeddings to be used for vectorization.
        """
        if isinstance(embeddings, str):
            embeddings = OllamaEmbeddings(model=embeddings)

        client = QdrantClient(":memory:", check_compatibility=False)

        if (
            not client.collection_exists(QDRANT_COLLECTION_NAME)
            or recreate_collection
        ):
            print(f"Creating Qdrant collection: {QDRANT_COLLECTION_NAME}")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION, distance=Distance.COSINE
                ),
            )

        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embeddings,
        )

    def add_documents(self, documents: list[Document], ids: list[int] = None):
        """
        Add documents to the vector store.
        @param documents: A list of Document instances to be added to the vector store.
        """
        self.vector_store.add_documents(documents, ids=ids)

    def similarity_search_with_score(self, query: str, k: int = 5):
        """
        Perform a similarity search on the vector store and then rerank the results.
        @param query: The query string to search for similar documents.
        @param k: The number of top similar documents to retrieve.
        @return: A list of tuples containing Document instances and their similarity scores.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def retrieve(self, query: str, k: int = 5):
        """
        Retrieve documents based on the query.
        @param query: The query string to search for similar documents.
        @param k: The number of top similar documents to retrieve.
        @return: A list of Document instances that are most similar to the query.
        """
        docs = self.similarity_search_with_score(query, k=k * 3)
        reranked_docs = rerank_documents(
            query, [doc for doc, _ in docs], top_k=k
        )
        return reranked_docs


class RetrieverSingleton(Retriever):
    """Singleton class for the Retriever."""

    _instance = None

    def __new__(cls, embeddings: Embeddings):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(embeddings)
        return cls._instance


def reranker(query: str, documents: list[str], top_k: int = 10) -> list[dict]:
    """
    Rerank a list of documents based on their relevance to the query using a CrossEncoder model.
    """

    rankings = model.rank(query, documents, top_k=top_k)
    return rankings


def rerank_documents(
    query: str, documents: list[Document], top_k: int = 10
) -> list:
    """
    Reorder documents by reranker relevance to the query.
    Skips the model entirely when the candidate pool is tiny.
    Returns up to top_k documents.
    """
    if len(documents) <= SMALL_SET_SKIP:
        return documents[:top_k]

    contents = [d.page_content for d in documents]

    ranks = reranker(query, contents, top_k=top_k)
    ranks.sort(key=lambda x: x["score"], reverse=True)

    sorted_idx = [x["corpus_id"] for x in ranks if x["score"] >= 0]

    return [documents[idx] for idx in sorted_idx][:top_k]


def rerank_docs(
    query: str, documents: list[Document], top_k: int = 10
) -> list[str]:
    """
    Rerank a list of documents based on their relevance to the query using a CrossEncoder model.
    Returns a list of dictionaries with 'corpus_id' and 'score'.
    """
    return rerank_documents(query, documents, top_k=top_k)
