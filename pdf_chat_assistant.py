import os
from typing import Optional, List
from langchain_ollama import ChatOllama
from langchain.schema import Document
from langchain.retrievers import BaseRetriever


def validate_pdf_path(file_path: str) -> bool:
    """
    Validate if the provided file path exists and points to a PDF file.

    Args:
        file_path (str): Path to the file to validate

    Returns:
        bool: True if file exists and is a PDF, False otherwise
    """
    return file_path.endswith('.pdf') and os.path.exists(file_path)


def initialize_retriever() -> BaseRetriever:
    """
    Initialize retriever with user-provided PDF file.
    Continuously prompts user until a valid PDF path is provided.

    Returns:
        BaseRetriever: Initialized retriever object for document processing
    """
    while True:
        file_path = input("Enter the path to your PDF file: ").strip()
        if validate_pdf_path(file_path):
            print(f"\nLoading and processing {file_path}...")
            return retriever(file_path)
        print("Invalid file path. Please make sure the file exists and is a PDF.")


def chat_llm() -> ChatOllama:
    """
    Initialize and configure the ChatOllama model with predefined settings.

    Returns:
        ChatOllama: Configured chat model instance
    """
    return ChatOllama(
        model=Config.ChatBot.NAME,
        temperature=Config.ChatBot.TEMPERATURE,
        keep_alive=-1,
        verbose=False
    )


def run_chat_session(retriever: BaseRetriever, llm: ChatOllama) -> None:
    """
    Run an interactive chat session using the initialized retriever and language model.

    Args:
        retriever (BaseRetriever): Document retriever for context lookup
        llm (ChatOllama): Language model for generating responses
    """
    print("Processing complete. You can now ask questions about the document.")
    print("Type 'bye' to exit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'bye':
            print("Assistant: Goodbye!")
            break

        # Retrieve relevant documents based on user input
        relevant_docs: List[Document] = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # Format prompt with context and user question
        prompt = Config.ChatBot.QA_PROMPT.format(
            context=context,
            question=user_input
        )

        # Generate and display response
        response = llm.invoke(prompt)
        print("\nAssistant:", response.content)


if __name__ == "__main__":
    # Initialize components
    retriever = initialize_retriever()
    chatbot = chat_llm()

    # Start interactive chat session
    run_chat_session(retriever=retriever, llm=chatbot)