import csv
from datetime import datetime
from pathlib import Path
import os
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from langchain_core.documents import Document

from mrs.clientsManager import ClientsManager
from mrs import constants
from mrs import Retriever


def clean_text(text):
    # Remove non-alphanumeric characters and replace spaces with underscores
    text = "".join(c if c.isalnum() else "_" for c in text)
    return text


def evaluator(dataset: str, retriever: Retriever, description: str = ""):
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

    retriever.add_documents(documents, qdrant_ids)

    # -------------------------
    # Retrieve
    # -------------------------
    results = {}

    for query_id, query in queries.items():

        docs_with_scores = retriever.similarity_search_with_score(
            query,
            k=10,
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

    mrr = evaluator.evaluate_custom(
        qrels,
        results,
        [1, 3, 5, 10],
        metric="mrr",
    )

    print("\nNDCG")
    print(ndcg)

    print("\nMAP")
    print(_map)

    print("\nRecall")
    print(recall)

    print("\nPrecision")
    print(precision)

    ### If you want to save your results and runfile (useful for reranking)
    results_dir = (
        Path(__file__).parent.absolute() / "results" / clean_text(dataset)
    )

    os.makedirs(results_dir, exist_ok=True)

    #### Save the evaluation runfile & results
    util.save_runfile(
        results_dir / f"{constants.EMBEDDING_MODEL}.run.trec", results
    )
    util.save_results(
        results_dir / f"{constants.EMBEDDING_MODEL}.json",
        ndcg,
        _map,
        recall,
        precision,
        mrr,
    )

    #### Append run to CSV log for improvement tracking
    metrics = {
        **ndcg,
        **_map,
        **recall,
        **precision,
        **mrr,
    }

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": constants.EMBEDDING_MODEL,
        "description": description,
        **metrics,
    }

    csv_path = results_dir / f"{clean_text(dataset)}.csv"

    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
