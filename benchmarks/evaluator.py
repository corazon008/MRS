from pathlib import Path
import os
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from langchain_core.documents import Document

from mrs.clientsManager import ClientsManager
from mrs import constants


def evaluator(dataset: str, clientManager: ClientsManager):
    collection_name = f"{dataset}_collection_{constants.EMBEDDING_MODEL}"

    # -------------------------
    # Download BEIR dataset
    # -------------------------
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/"
        f"thakur/BEIR/datasets/{dataset}.zip"
    )

    out_dir = Path(__file__).parent.parent.absolute() / "datasets"

    data_path = util.download_and_unzip(
        url,
        out_dir,
    )

    # -------------------------
    # Load BEIR
    # -------------------------
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split="test"
    )

    # -------------------------
    # Index corpus
    # -------------------------
    documents = []
    qdrant_ids = []

    for doc_id, data in corpus.items():
        text = f"{data['title']}\n\n{data['text']}"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "beir_doc_id": doc_id,
                },
            )
        )

        qdrant_ids.append(int(doc_id))

    clientManager.initialize_vector_store(
        collection_name=collection_name,
    )
    # If the collection already exists, we skip adding documents to avoid duplicates.
    if clientManager.qdrant_client.count(
        collection_name=collection_name
    ) != len(documents):
        clientManager.initialize_vector_store(
            collection_name=collection_name,
            force=True,
        )

        clientManager.vector_store.add_documents(
            documents,
            ids=qdrant_ids,
        )

    # -------------------------
    # Retrieve
    # -------------------------
    results = {}

    for query_id, query in queries.items():

        docs_with_scores = (
            clientManager.vector_store.similarity_search_with_score(
                query,
                k=10,
            )
        )

        results[query_id] = {
            doc.metadata["beir_doc_id"]: score
            for doc, score in docs_with_scores
        }

    # -------------------------
    # Evaluate
    # -------------------------

    evaluator = EvaluateRetrieval()

    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels,
        results,
        [1, 3, 5, 10],
    )

    print("\nNDCG")
    print(ndcg)

    print("\nMAP")
    print(_map)

    print("\nRecall")
    print(recall)

    print("\nPrecision")
    print(precision)

    # Save results to a file
    results_dir = Path(__file__).parent.parent.absolute() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = (
        results_dir / f"{dataset}_{constants.EMBEDDING_MODEL}_results.txt"
    )
    with open(results_file, "w") as f:
        f.write("NDCG\n")
        f.write(str(ndcg) + "\n\n")

        f.write("MAP\n")
        f.write(str(_map) + "\n\n")

        f.write("Recall\n")
        f.write(str(recall) + "\n\n")

        f.write("Precision\n")
        f.write(str(precision) + "\n\n")
