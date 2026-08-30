from mrs.retrieval import Retriever

from evaluator import evaluator, clean_text
from mrs import constants

if __name__ == "__main__":

    dataset = "quora"

    constants.EMBEDDING_MODEL = "nomic-embed-text"

    constants.QDRANT_COLLECTION_NAME = (
        f"{dataset}_collection_{clean_text(constants.EMBEDDING_MODEL)}"
    )

    retriever = Retriever(embeddings=constants.EMBEDDING_MODEL)

    evaluator(
        dataset=dataset,
        retriever=retriever,
        description="baseline nomic-embed-text",
    )

    dataset = "scifact"

    constants.QDRANT_COLLECTION_NAME = (
        f"{dataset}_collection_{clean_text(constants.EMBEDDING_MODEL)}"
    )

    retriever = Retriever(embeddings=constants.EMBEDDING_MODEL, test_mode=True)

    evaluator(
        dataset=dataset,
        retriever=retriever,
        description="baseline nomic-embed-text",
    )
