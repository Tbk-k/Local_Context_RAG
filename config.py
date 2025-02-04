class Config:
    class Preprocessing:
        CHUNK_SIZE = 1024
        CHUNK_OVERLAP = 64
        LLM = 'llama3.2'
        CONTEXT_PROMPT = '''
        You're an expert in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document.

        Here is the document:
        <document>
        {doc}
        </document>

        Here is the chunk we want to situate within the whole document:
        <chunk>
        {chunk}
        </chunk>

        Provide a concise context (2–3 sentences) for this chunk, considering the following guidelines:
        1. Identify the main topic or concept discussed in the chunk.
        2. Mention any relevant information or comparisons from the broader document context.
        3. If applicable, note how this information relates to the overall theme or purpose of the document.
        4. Include any key figures, dates, or percentages that provide important context.
        5. Do not use phrases like "This chunk discusses" or "This section provides." Instead, directly state the context.

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.

        Context:
        '''

        EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
        CONTEXT_CHUNK_RANGE = 5
        N_SEMANTIC_RESULTS = 5
        RERANKER = 'ms-marco-MiniLM-L-12-v2'

    class ChatBot:
        N_COTEXT_RESULT = 3
        NAME = 'deepseek-r1:14b'
        TEMPETARURE = 0.0
        QA_PROMPT = '''
        Use the following context to provide a precise and accurate answer to the user's question. 
        If the context does not contain sufficient information to answer the question, clearly state that you 
        cannot find the answer in the provided text.

        Context:
        {context}

        Question: {question}

        Your response should:
        - Be based strictly on the given context
        - Provide clear and direct information
        - Cite specific details from the context when possible
        - If no relevant information exists, say "I cannot find an answer in the provided text"
        '''
