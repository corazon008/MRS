from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pathlib import Path
import os
import subprocess

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from . import constants

ROOT_DIR = Path(__file__).parent.parent.parent


class ClientsManager:
    """Singleton class to manage Ollama models and Qdrant collection."""

    def __init__(self):
        """
        Initialize the ClientsManager.
        """
        self.wait_for_qdrant()

        self.ollama_client = Client()
        self.qdrant_client = QdrantClient()

        # Ensure models are installed
        self.ollama_client.pull(constants.LLM_MODEL)
        self.ollama_client.pull(constants.EMBEDDING_MODEL)

    def initialize_vector_store(
        self, collection_name: str = None, force: bool = False
    ):
        """
        Initialize the Qdrant vector store and retriever.
        @param collection_name: Optional name of the Qdrant collection. If not provided, a default name is used.
        @param force: If True, forces the creation of the collection even if it already exists.
        """
        self.collection_name = (
            collection_name or constants.QDRANT_COLLECTION_NAME
        )
        self.qdrant_create_collection(self.collection_name, force=force)

        self.embeddings = OllamaEmbeddings(
            model=constants.EMBEDDING_MODEL,
        )
        print(
            f"Initialized Ollama embeddings with model: {constants.EMBEDDING_MODEL}"
        )

        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        # Initialize retriever
        self.retriever = VectorStoreRetriever(
            vectorstore=self.vector_store,
        )

    def wait_for_qdrant(self, timeout=30):
        """Wait for the Qdrant server to be ready."""
        import time
        import requests

        start_time = time.time()
        while True:
            try:
                response = requests.get("http://localhost:6333/collections")
                if response.status_code == 200:
                    break
            except requests.ConnectionError:
                pass

            if time.time() - start_time > timeout:
                raise TimeoutError("Qdrant server did not start in time.")
            time.sleep(1)

    def qdrant_create_collection(
        self, collection_name: str, force: bool = False
    ):
        """Create a Qdrant collection if it doesn't exist."""
        if (
            not self.qdrant_client.collection_exists(collection_name)
            and not force
        ):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=constants.EMBEDDING_DIMENSION, distance=Distance.COSINE
                ),
            )
        else:
            try:
                self.qdrant_client.delete_collection(collection_name)
            except Exception as e:
                print(
                    f"Warning: Could not delete collection {collection_name}: {e}"
                )
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=constants.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE,
                ),
            )
