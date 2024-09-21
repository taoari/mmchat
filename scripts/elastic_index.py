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
    parser.add_argument('--es-url', default='http://localhost:9200',
            help='Elasticsearch url')
    parser.add_argument('--embeddings', default=None,
            help='Embeddings')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    from langchain_elasticsearch import ElasticsearchStore
    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings()

    vectordb = ElasticsearchStore(
        args.collection_name, embedding=embeddings, es_url=args.es_url
    )
    pdfs = sorted(glob.glob(os.path.join(args.folder, '**', '*.pdf'), recursive=True))

    for i, fname in enumerate(pdfs):
        print(f'Indexing: {fname}')
        index_file(vectordb, fname)



