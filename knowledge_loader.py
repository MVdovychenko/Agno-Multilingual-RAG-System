from agno.vectordb.chroma import ChromaDb                  # manages loading/refreshing a PDF corpus + wiring it to a vector DB
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader # extracts text from PDFs (page-wise, supports simple chunking)
from agno.agent import Agent                               # core Agno agent class (LLM + tools/knowledge + policies)
from agno.models.openai import OpenAIChat                  # Agno model adapter for OpenAI chat models (e.g., gpt-4o-mini).
from agno.document.chunking.agentic import AgenticChunking # advanced chunking strategy that uses an LLM to chunk documents based on content and semantics


# Paths for knowledge base input and vector DB storage
KNOWLEDGE_DIR = "/data/knowledge"
COLLECTION_NAME = "documentation"
CHROMA_DIR = "/app/chroma_store"


# Persistent Chroma client (keeps embeddings across container restarts)
chroma_client=ChromaDb(collection=COLLECTION_NAME, persistent_client=True, path=CHROMA_DIR)


# Knowledge base: abstracts parsing, chunking, embedding, and storage
pdf_knowledge_base = PDFKnowledgeBase(
    path=KNOWLEDGE_DIR,
    vector_db=chroma_client,
    num_documents=5,               # top-k chunks returned for retrieval
    reader=PDFReader(chunk=True),  # page-wise reading with simple chunking per page
    chunking_strategy=AgenticChunking(model=OpenAIChat(id="gpt-4o-mini")), # let a small LLM pick semantic breakpoints
)


# Factory function that creates and returns a RAG agent (PDFs as knowledge base)
def pdf_agent(): 
    """Load the PDF knowledge base and create an agent."""
    pdf_knowledge_base.load(recreate=False)

    return  Agent(
        description="You are an AI assistant who's going to use appropriate tools to answer the user's questions.",
        model=OpenAIChat(id="gpt-4o-mini"), # abstracts raw OpenAI API calls
        markdown=True,                      # format responses as Markdown
        knowledge=pdf_knowledge_base,       # attach knowledge base so retrieval is available
        search_knowledge=True,              # RAG: retrieve top-k before answering
        instructions=[
            "First, detect the language of the input question.",
            "If the knowledge base is in a different language, translate the information internally, but respond in the user's language or the language that the user asked you to use it.",
            "Always cite the source title and page number for every fact you provide.",
            "If you cannot find enough information in the knowledge base, politely say that you do not know.",
            "Never invent information or speculate.",
        ],
        # expected_output="a how to guide in the language of the asked question",
    )