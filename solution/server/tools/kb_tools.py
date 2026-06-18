from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from server.dependencies import CHROMA_PATH

mcp = FastMCP("kb-tools")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory=f"{CHROMA_PATH}/knowledge",
    embedding_function=embeddings,
    collection_name="knowledge_articles",
)


@mcp.tool()
def search_knowledge_base(
    query: str,
    account_id: str,
    top_k: int = 3,
) -> dict:
    """
    Performs semantic search over the knowledge base for a given account.

    Embeds the query, retrieves top-k chunks from ChromaDB
    filtered by account_id, and returns ranked results.

    Args:
        query:      The customer's message or search query
        account_id: Scopes the search to this account's articles
        top_k:      Number of results to return (default 3, max 10)

    Returns:
        dict with 'results' list, each containing article_id,
        title, content, tags, and relevance_score.
    """
    top_k = min(top_k, 10)

    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter={"account_id": account_id},
    )

    results = [
        {
            "article_id":      doc.metadata.get("article_id", ""),
            "title":           doc.metadata.get("title", ""),
            "content":         doc.page_content,
            "tags":            doc.metadata.get("tags"),
            "relevance_score": round(score, 4),
        }
        for doc, score in docs_with_scores
    ]

    return {
        "query":       query,
        "total_found": len(results),
        "results":     results,
    }
