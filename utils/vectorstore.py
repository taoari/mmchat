import os
import warnings
import hashlib
from collections import defaultdict
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml import YAML

def repr_str(dumper: RoundTripRepresenter, data: str):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml = YAML()
yaml.representer.add_representer(str, repr_str)


def _get_hash(content, is_file=False):
    """Generate an MD5 hash for a string or a file."""
    if is_file:
        with open(content, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    return hashlib.md5(content.encode('utf8')).hexdigest()


def _initialize_embedding():
    """Initialize the embedding function, using a custom endpoint if specified in the environment."""
    hf_endpoint = os.environ.get("HF_EMBEDDINGS_ENDPOINT")
    if hf_endpoint:
        from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
        return HuggingFaceEndpointEmbeddings(model=hf_endpoint)
    warnings.warn('Environment variable HF_EMBEDDINGS_ENDPOINT is not set, defaulting to HuggingFaceEmbeddings.')
    return HuggingFaceEmbeddings()


def _split_documents(pages, chunk_size):
    """Split documents into chunks if chunk_size is greater than 0."""
    if chunk_size > 0:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        return splitter.split_documents(pages)
    return pages


def _deduplicate_documents(docs, vectordb):
    """Remove documents that are already in the vector database."""
    ids = [_get_hash(doc.page_content) for doc in docs]
    existing_ids = set(vectordb.get()['ids'])
    return {_id: doc for _id, doc in zip(ids, docs) if _id not in existing_ids}


def _build_vs(fname, chunk_size=0, persist_directory=None, collection_name=None, max_pages=0, autogen_yaml=False, verbose=False):
    """Build a vector store from a PDF file."""
    loader = PyPDFLoader(fname)
    pages = loader.load()
    if autogen_yaml:
        yaml_fname = os.path.splitext(fname)[0] + '.autogen.yaml'
        with open(yaml_fname, 'w', encoding='utf-8') as f:
            _pages = [dict(metadata=page.metadata, page_content=page.page_content) for page in pages]
            # yaml.safe_dump(_pages, f)
            yaml.dump(_pages, f)

    docs = _split_documents(pages, chunk_size)

    if verbose:
        print(f'len(pages) = {len(pages)}')
        print(f'len(docs) = {len(docs)}')

    if max_pages > 0:
        docs = docs[:max_pages]
        if verbose:
            print(f'len(docs) for vs = {len(docs)}')

    # Initialize embeddings
    embedding = _initialize_embedding()

    # Set collection name based on file hash if not provided
    if collection_name is None:
        collection_name = _get_hash(fname, is_file=True)

    # Initialize Chroma vector store
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if verbose:
        print(f"vector db {vectordb._collection.name} has {vectordb._collection.count()} records: {fname}")

    # Deduplicate documents
    docs_dedup = _deduplicate_documents(docs, vectordb)

    # Update vector store with new documents
    if docs_dedup:
        vectordb = Chroma.from_documents(
            documents=list(docs_dedup.values()),
            ids=list(docs_dedup.keys()),
            collection_name=collection_name,
            embedding=embedding,
            persist_directory=persist_directory,
        )

    if persist_directory:
        vectordb.persist()

    if verbose:
        print(f"updated vector db {vectordb._collection.name} has {vectordb._collection.count()} records: {fname}")

    return vectordb


def _load_vs(persist_directory=None):
    """Load an existing vector store."""
    embedding = HuggingFaceEmbeddings()
    return Chroma(embedding_function=embedding, persist_directory=persist_directory)


def _build_vs_collection(folder, collection_name, chunk_size=0, persist_directory=None, max_pages=0, autogen_yaml=False, verbose=False):
    """Build a vector store for all PDFs in a folder."""
    if not os.path.exists(folder):
        return None

    pdfs = sorted(glob.glob(os.path.join(folder, '**', '*.pdf'), recursive=True))
    vectordb = None

    for pdf in pdfs:
        vectordb = _build_vs(
            fname=pdf,
            chunk_size=chunk_size,
            persist_directory=persist_directory,
            collection_name=collection_name,
            max_pages=max_pages,
            autogen_yaml=autogen_yaml,
            verbose=verbose
        )

    return vectordb


if __name__ == '__main__':
    load_dotenv()
    vectordb = _build_vs_collection('data/collections/audio2face', 'audio2face', autogen_yaml=True, verbose=True)