from langchain_ollama import ChatOllama
from pypdfium2 import PdfDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional
from langchain_core.documents import Document
import pandas as pd
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from flashrank import Ranker
from config import Config


def _data_loader(pdf_path: str) -> List[str]:
    """
    Extract text content from each page of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[str]: List of text content from each page
    """
    pdf = PdfDocument(pdf_path)
    pages = []
    for page in pdf:
        content = page.get_textpage()
        text = content.get_text_bounded()
        pages.append(text)
    return pages


# Initialize text splitter with configuration parameters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.Preprocessing.CHUNK_SIZE,
    chunk_overlap=Config.Preprocessing.CHUNK_OVERLAP
)


def _llm(model: str = 'llama3.2') -> ChatOllama:
    """
    Initialize a Language Learning Model with specified parameters.

    Args:
        model (str): Name of the model to use

    Returns:
        ChatOllama: Configured language model instance
    """
    return ChatOllama(
        model=model,
        temperature=0,  # Deterministic output
        keep_alive=-1  # Keep the model alive indefinitely
    )


def _create_context(llm: ChatOllama, doc: str, chunk: str) -> str:
    """
    Generate context for a text chunk using the language model.

    Args:
        llm (ChatOllama): Language model instance
        doc (str): Document text
        chunk (str): Text chunk to create context for

    Returns:
        str: Generated context
    """
    msg = Config.Preprocessing.CONTEXT_PROMPT.format(doc=doc, chunk=chunk)
    response = llm.invoke(msg)
    return response.content


def _build_context_text(
        total_chunks: int,
        current_chunk_idx: int,
        context_chunk_range: int = Config.Preprocessing.CONTEXT_CHUNK_RANGE,
        chunks: List[Document] = []
) -> str:
    """
    Build context text by combining surrounding chunks.

    Args:
        total_chunks (int): Total number of chunks
        current_chunk_idx (int): Index of current chunk
        context_chunk_range (int): Range of chunks to include in context
        chunks (List[Document]): List of document chunks

    Returns:
        str: Combined context text
    """
    start = max(0, current_chunk_idx - context_chunk_range)
    end = min(total_chunks, current_chunk_idx + context_chunk_range + 1)
    context_text = ''
    for i in range(start, end):
        context_text += chunks[i].page_content
    return context_text


def _create_chunks(pages: List[str]) -> List[Document]:
    """
    Create document chunks with context using the language model.

    Args:
        pages (List[str]): List of text pages to process

    Returns:
        List[Document]: List of processed document chunks with context
    """
    llm = _llm()
    docs = splitter.create_documents(pages)
    chunks = splitter.split_documents(docs)
    total_chunks = len(chunks)
    context_chunks = []
    for idx, chunk in enumerate(chunks):
        context_text = _build_context_text(total_chunks=total_chunks, current_chunk_idx=idx, chunks=chunks)
        print(f'Processing chunk {idx + 1} of {total_chunks}')
        context_chunk = _create_context(llm, doc=context_text, chunk=chunk.page_content)
        context_chunks.append(Document(page_content=context_chunk, metadata=chunk.metadata or {}))
    return context_chunks


def save_context_chunk_to_csv(output_file: str, chunks: List[str]) -> None:
    """
    Save contextual chunks to a CSV file.

    Args:
        output_file (str): Path to save the CSV file
        chunks (List[str]): List of chunks to save
    """
    df = pd.DataFrame(chunks)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f'Contextual chunks saved to {output_file}')


def _create_embeddings() -> FastEmbedEmbeddings:
    """
    Initialize embedding model with configured parameters.

    Returns:
        FastEmbedEmbeddings: Configured embedding model
    """
    return FastEmbedEmbeddings(
        model=Config.Preprocessing.EMBEDDING_MODEL
    )


def _create_compressor() -> FlashrankRerank:
    """
    Create a reranking compressor for document retrieval.

    Returns:
        FlashrankRerank: Configured reranking compressor
    """
    compressor = FlashrankRerank(
        model=Config.Preprocessing.RERANKER,  # Model for reranking
        top_n=3  # Number of top results to keep
    )
    return compressor


def retriever(file_path: str) -> ContextualCompressionRetriever:
    """
    Create a document retrieval system with semantic search and keyword-based retrieval.

    Args:
        file_path (str): Path to the PDF file to process

    Returns:
        ContextualCompressionRetriever: Configured retrieval system combining semantic
        and keyword-based search with reranking
    """
    # Load pages from the PDF
    pages = _data_loader(file_path)

    # Create chunks with context
    chunks = _create_chunks(pages)

    # Create a semantic retriever using embeddings
    semantic_retriever = InMemoryVectorStore.from_documents(
        chunks,
        _create_embeddings()
    ).as_retriever(search_kwargs={'k': Config.Preprocessing.N_SEMANTIC_RESULTS})

    # Create a BM25 retriever for keyword-based search
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.Preprocessing.N_SEMANTIC_RESULTS

    # Combine semantic and BM25 retrievers into an ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # Weighted combination
    )

    # Apply contextual compression to the ensemble retriever
    return ContextualCompressionRetriever(
        base_compressor=_create_compressor(),
        base_retriever=ensemble_retriever
    )