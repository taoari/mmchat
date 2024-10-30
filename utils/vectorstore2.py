import os, glob
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def _get_hash(content: str, is_file: bool = False) -> str:
    """Generate an MD5 hash for a string or a file."""
    if is_file:
        with open(content, "rb") as file:
            file_hash = hashlib.md5()
            while chunk := file.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    return hashlib.md5(content.encode('utf8')).hexdigest()


def index_file(vectordb, fname):

    loader = PyPDFLoader(fname)
    pages = loader.load()

    docs = pages
    ids = [_get_hash(doc.page_content) for doc in docs]

    print(f'\tlen(pages) = {len(pages)}, len(docs) = {len(docs)}')

    vectordb.add_documents(documents=docs, ids=ids)


def index_docs(vectordb, docs):
    ids = [_get_hash(doc.page_content) for doc in docs]
    print(f'\tlen(docs) = {len(docs)}')
    for id, doc in zip(ids, docs):
        if 'source' not in doc.metadata:
            doc.metadata['source'] = id
    vectordb.add_documents(documents=docs, ids=ids)


def get_embeddings(embeddings_conn_str):
    if embeddings_conn_str:
        endpoint, model_id = embeddings_conn_str.split(';')
        # NOTE: infinity embeddings are compatible with OpenAI embeddings, 
        # theoretically should be able to use OpenAIEmbeddings
        from langchain_community.embeddings import InfinityEmbeddings
        embeddings = InfinityEmbeddings(model=model_id, infinity_api_url=f'{endpoint}/v1')
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()
    return embeddings


def get_vectordb(vectorstore, collection_name, embeddings_conn_str=None):
    embeddings = get_embeddings(embeddings_conn_str)

    if vectorstore == 'chroma':
        from langchain_chroma import Chroma

        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

    elif vectorstore == 'elasticsearch':
        from langchain_elasticsearch import ElasticsearchStore

        ES_URL = "http://localhost:9200"

        vectordb = ElasticsearchStore(
            collection_name, embedding=embeddings, es_url=ES_URL
        )

    elif vectorstore == 'redis':
        from langchain_redis import RedisConfig, RedisVectorStore

        REDIS_URL = "redis://localhost:6379"

        config = RedisConfig(
            index_name=collection_name,
            redis_url=REDIS_URL,
            metadata_schema=[
                {"name": "source", "type": "text"},
                {"name": "page", "type": "numeric"},
            ],
        )

        vectordb = RedisVectorStore(embeddings, config=config)

    elif vectorstore == 'pgvector':
        from langchain_core.documents import Document
        from langchain_postgres import PGVector
        from langchain_postgres.vectorstores import PGVector

        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"  # Uses psycopg3!

        vectordb = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

    else:
        raise ValueError(f"Invalid vector store: {vectorstore}")
    return vectordb


def _print_vectordb_info(vectordb):
    if 'chroma' in vectordb.__class__.__name__.lower():
        print(f"\tvector db {vectordb._collection.name} has {vectordb._collection.count()} records")


def build_vectordb(vectorstore, collection_name, folder_or_file, embeddings_conn_str=None):
    vectordb = get_vectordb(vectorstore, collection_name, embeddings_conn_str)
    if os.path.isdir(folder_or_file):
        pdfs = sorted(glob.glob(os.path.join(folder_or_file, '**', '*.pdf'), recursive=True))

        for i, fname in enumerate(pdfs):
            print(f'Indexing {i} of {len(pdfs)}: {fname}')
            index_file(vectordb, fname)
            _print_vectordb_info(vectordb)
    elif folder_or_file.endswith('.json'):
        import json
        with open(folder_or_file) as f:
            _docs = json.load(f)

        docs = []
        for _doc in _docs:
            page_content = _doc['content'] if 'content' in _doc else _doc['text']
            metadata = {k: v for k, v in _doc.items() if k not in ['content', 'text']}
            docs.append(Document(page_content=page_content, metadata=metadata))
        index_docs(vectordb, docs)

    return vectordb

def parse_args():
    """
    Parses command-line arguments and returns them as an argparse.Namespace object.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Elastic Index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-f', '--folder_or_file', default='data/collections/default',
            help='Folder or JSON file')
    parser.add_argument('-c', '--collection-name', default='mycollection',
            help='Collection name')
    parser.add_argument('-vs', '--vectorstore', default='chroma',
            help='Vector store')
    parser.add_argument('--embeddings', default=None, type=str, 
        help='embeddings (format: "<endpoint>;<model_id>")')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    build_vectordb(args.vectorstore, args.collection_name, args.folder_or_file, 
            embeddings_conn_str=args.embeddings)