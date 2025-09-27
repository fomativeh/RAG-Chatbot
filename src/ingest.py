import argparse
import os
import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from utils import get_or_create_collection, get_chunking, get_models, get_openrouter_client
from metadata import infer_file_metadata, extract_chunk_section


def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


def ingest_directory(data_dir: str) -> None:
    collection = get_or_create_collection()
    chunk_size, overlap, _ = get_chunking()

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {data_dir}. Place files and rerun.")
        return

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        print(f"Ingesting {path} ...")
        docs = load_pdf(path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        split_docs = splitter.split_documents(docs)

        file_md = infer_file_metadata(filename)

        ids = [str(uuid.uuid4()) for _ in split_docs]
        texts = [d.page_content for d in split_docs]
        metadatas = []
        for i, d in enumerate(split_docs):
            md = dict(d.metadata or {})
            md.update({"source": filename, "chunk_index": i})
            # Enrich with legal metadata
            md.update(file_md)
            # Try to extract section/article info from the chunk text
            sec = extract_chunk_section(d.page_content)
            md.update(sec)
            metadatas.append(md)

        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        print(f"Added {len(texts)} chunks from {filename}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma")
    parser.add_argument("--path", default="data", help="Directory containing PDFs")
    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    ingest_directory(args.path)


if __name__ == "__main__":
    main()


