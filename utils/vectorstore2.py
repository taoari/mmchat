import os, glob
import hashlib
from langchain_community.document_loaders import PyPDFLoader


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

    vectordb.add_documents(documents=docs, ids=ids)


def get_vectordb(args):
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings()

    if args.vectorstore == 'elasticsearch':
        from langchain_elasticsearch import ElasticsearchStore

        ES_URL = "http://localhost:9200"

        vectordb = ElasticsearchStore(
            args.collection_name, embedding=embeddings, es_url=ES_URL
        )

    elif args.vectorstore == 'redis':
        from langchain_redis import RedisConfig, RedisVectorStore

        REDIS_URL = "redis://localhost:6379"

        config = RedisConfig(
            index_name=args.collection_name,
            redis_url=REDIS_URL,
            metadata_schema=[
                {"name": "source", "type": "text"},
                {"name": "page", "type": "numeric"},
            ],
        )

        vectordb = RedisVectorStore(embeddings, config=config)

    elif args.vectorstore == 'pgvector':
        from langchain_core.documents import Document
        from langchain_postgres import PGVector
        from langchain_postgres.vectorstores import PGVector

        # See docker command above to launch a postgres instance with pgvector enabled.
        connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"  # Uses psycopg3!

        vectordb = PGVector(
            embeddings=embeddings,
            collection_name=args.collection_name,
            connection=connection,
            use_jsonb=True,
        )

    else:
        raise ValueError(f"Invalid vector store: {args.vectorstore}")
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
    parser.add_argument('--folder', default='data/collections/default',
            help='Folder')
    parser.add_argument('-c', '--collection-name', default='mycollection',
            help='Collection name')
    parser.add_argument('-vs', '--vectorstore', default='elasticsearch',
            help='Vector store')
    parser.add_argument('--embeddings', default=None,
            help='Embeddings')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    vectordb = get_vectordb(args)
    pdfs = sorted(glob.glob(os.path.join(args.folder, '**', '*.pdf'), recursive=True))

    for i, fname in enumerate(pdfs):
        print(f'Indexing: {fname}')
        index_file(vectordb, fname)