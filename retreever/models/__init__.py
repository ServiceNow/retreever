from retreever.models.retreever import ReTreever
from retreever.models.faiss_inner_product import FaissInnerProductRetriever

__all__ = ["KNOWN_MODEL_TYPE"]

KNOWN_MODEL_TYPE = {
    "retreever": ReTreever,
    "faiss_inner_product": FaissInnerProductRetriever,
    "faiss_ivf": FaissInnerProductRetriever,
}
