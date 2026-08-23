import os
import pathlib
import logging

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from langchain_core.documents import Document

from mrs.clientsManager import ClientsManager
from mrs import constants

from evaluator import evaluator
from mrs import constants

dataset = "quora"

clientManager = ClientsManager()

constants.EMBEDDING_MODEL = "qwen3-embedding:0.6b"

evaluator(dataset=dataset, clientManager=clientManager)

dataset = "scifact"

clientManager = ClientsManager()

evaluator(dataset=dataset, clientManager=clientManager)
