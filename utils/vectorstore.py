import os
import bs4
from functools import cache
from langchain_openai import ChatOpenAI
from config import config, secrets
from config.config import LLM_ENDPOINTS


@cache
def get_llm(chat_model="gpt-4o-mini"):
    assert chat_model in LLM_ENDPOINTS

    model_name = LLM_ENDPOINTS[chat_model]["model_name"]

    base_url = LLM_ENDPOINTS[chat_model]["base_url"]
    provider = LLM_ENDPOINTS[chat_model]["provider"]
    api_key = {
        "onprem": "-",
        "openai": secrets.OPENAI_API_KEY,
        "openrouter": secrets.OPENROUTER_API_KEY,
    }[provider]

    llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key)
    return llm

    # client = OpenAI(api_key=api_key, base_url=base_url)

    # if model_name in config.LLM_ENDPOINTS:
    #     model_name_hf = config.LLM_ENDPOINTS[model_name]["model_name"]
    #     base_url = config.LLM_ENDPOINTS[model_name]["base_url"]
    #     llm = ChatOpenAI(model=model_name_hf, base_url=base_url)
    # else:
    #     llm = ChatOpenAI(model=model_name)
    # return llm


@cache
def get_embeddings(model_name="sentence-transformers/all-mpnet-base-v2"):
    if model_name in config.EMBED_ENDPOINTS:
        base_url = config.EMBED_ENDPOINTS[model_name]["base_url"]
        from langchain_community.embeddings import InfinityEmbeddings

        embeddings = InfinityEmbeddings(
            model=model_name,
            infinity_api_url=base_url,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def get_vector_store(embeddings):
    from langchain_core.vectorstores import InMemoryVectorStore

    vector_store = InMemoryVectorStore(embeddings)
    return vector_store


def build_sample_vector_store(vector_store):
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)


def prebuild_sample_vector_store():
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)
    build_sample_vector_store(vector_store)
    return vector_store


def get_vectordb(
    vs_type, collection_name, model_name=None, pre_delete_collection=False
):
    "Get vectordb."
    embeddings = get_embeddings(model_name)

    if vs_type == "inmemory":
        from langchain_core.vectorstores import InMemoryVectorStore

        vectordb = InMemoryVectorStore(embeddings)

    elif vs_type == "pgvector":
        from langchain_postgres.vectorstores import PGVector

        vectordb = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=secrets.PGVECTOR_CONNECTION_STR,
            use_jsonb=True,
            pre_delete_collection=pre_delete_collection,  # delete collection if exists
        )

    elif vs_type == "elasticsearch":
        from langchain_elasticsearch import ElasticsearchStore

        ES_URL = os.getenv("ES_URL", "http://localhost:9200")
        vectordb = ElasticsearchStore(
            collection_name, embedding=embeddings, es_url=ES_URL
        )

    else:
        raise ValueError(f"Invalid vector store type: {vs_type}")
    return vectordb


from typing import List, Tuple, Any


class MultiCollectionsVectorStore:
    def __init__(self, vectorstores: List[Any], weights: List[float]):
        if weights is None or not weights:
            weights = [1.0] * len(vectorstores)
        assert len(vectorstores) == len(weights), (
            "Number of vectorstores and weights must match."
        )
        self.vectorstores = vectorstores
        self.weights = weights

    def similarity_search_with_relevance_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Any, float]]:
        all_results = []

        for vs, weight in zip(self.vectorstores, self.weights):
            results = vs.similarity_search_with_relevance_scores(
                query, k
            )  # Fetch more to refine
            weighted_results = [(doc, score * weight) for doc, score in results]
            all_results.extend(weighted_results)

        # Sort by weighted score in descending order
        all_results.sort(key=lambda x: x[1], reverse=True)

        return all_results[:k]
