
# GIST:
# 1) Creates an Embedding model (GeminiEmbedder) to convert text -> vectors
# 2) Connects to Qdrant (vector DB) and binds the embedder to it
# 3) Wraps Qdrant inside Agno Knowledge so Agents can search your paper
# 4) Builds a small multi-agent Team:
#       - QA Agent: answers factual questions using retrieved chunks
#       - Summary Agent: writes structured summaries (problem/approach/results)
# 5) Exposes two clean functions for the rest of the app:
#       - ingest_pdf(): store a PDF into Qdrant collection
#       - ask_team(): answer a question using the team + knowledge base
# -------------------------------------------------------------------

from agno.agent import Agent
from agno.team import Team
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.models.google import Gemini
from agno.knowledge.embedder.google import GeminiEmbedder

from .config import settings


def build_knowledge(collection: str) -> Knowledge:
    """
    Build the 'Knowledge' object for ONE paper collection.

    In our design, each ingested paper has its own Qdrant collection:
        collection = "paper_<paper_id>"

    Knowledge = Agno's wrapper that lets agents:
    - search the stored chunks (vector similarity / hybrid, depending on setup)
    - retrieve relevant context automatically when answering
    """

    # Embedder = turns text chunks into vectors using Gemini embeddings.
    # Needs GOOGLE_API_KEY so Google can authorize embedding calls.
    embedder = GeminiEmbedder(api_key=settings.GOOGLE_API_KEY)

    # Vector DB = Qdrant instance pointing to our running Qdrant server.
    # We bind the embedder here so "insert()" automatically embeds the PDF chunks.
    vector_db = Qdrant(
        collection=collection,      # Qdrant collection name (paper-specific)
        url=settings.QDRANT_URL,     # Qdrant server URL (usually local docker)
        embedder=embedder,           # embedding model used for indexing + search
    )

    # Knowledge connects the Agent framework to the underlying vector database.
    return Knowledge(vector_db=vector_db)


def build_team(collection: str) -> Team:
    """
    Build a multi-agent Team that can answer questions about ONE paper.

    Why a team (instead of a single agent)?
    - QA Agent: strict factual answers from retrieved evidence
    - Summary Agent: structured summaries (useful for "summarize this paper")
    """

    # Knowledge is paper-specific, so the team only sees one paper at a time.
    knowledge = build_knowledge(collection)

    # LLM used for generation (answer writing / summarization).
    # Uses Gemini model id and your API key.
    llm = Gemini(id="gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)

    # Agent 1: QA Agent (factual answers)
    # search_knowledge=True means the agent can query Qdrant to pull relevant chunks.
    qa_agent = Agent(
        name="Paper QA Agent",
        role="Answer questions strictly using the uploaded research paper.",
        model=llm,
        knowledge=knowledge,
        search_knowledge=True,   # enables retrieval (RAG behavior)
        markdown=True,           # formats output nicely for UI rendering
    )

    # Agent 2: Summary Agent (structured summaries)
    summary_agent = Agent(
        name="Paper Summary Agent",
        role="Summarize problem, approach, results, limitations and future work clearly.",
        model=llm,
        knowledge=knowledge,
        search_knowledge=True,
        markdown=True,
    )

    # Team: routes a user prompt to the best agent.
    # Members list controls which agents are available.
    team = Team(
        name="Research Copilot Team",
        members=[qa_agent, summary_agent],
        model=llm,  # team-level model (some frameworks use this for coordination)
        instructions=(
            "Route questions to the best agent. "
            "Use QA agent for factual queries and Summary agent for summaries."
        ),
    )

    return team


def ingest_pdf(path: str, collection: str):
    """
    Ingest a PDF into the knowledge base.

    What happens under the hood:
    - Agno parses the PDF
    - splits it into chunks
    - embeds each chunk using GeminiEmbedder
    - upserts vectors + metadata into Qdrant under the given collection
    """
    knowledge = build_knowledge(collection)
    knowledge.insert(path=path)  # triggers parsing + chunking + embedding + storage
    return {"status": "ingested"}


def ask_team(question: str, collection: str):
    """
    Ask a question about a specific paper collection.

    Steps:
    1) Build the team (agents + knowledge bound to Qdrant collection)
    2) team.run(question) performs retrieval (search_knowledge=True)
    3) LLM produces the final answer grounded in retrieved chunks
    """
    team = build_team(collection)
    response = team.run(question)

    # Some frameworks return an object with .content; fallback to string if not.
    return {"answer": getattr(response, "content", str(response))}
