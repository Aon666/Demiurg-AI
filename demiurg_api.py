from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg
from pgvector.psycopg import register_vector
import ollama
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

demiurg_api = FastAPI(
    title="Demiurg-AI API",
    description="Local-First RAG Assistant for TTRPG Worldbuilding & Lore Management",
    version="1.0"
)

DB_PARAMS = "host=localhost port=5432 dbname=demiurg user=postgres password=postgrespassword"

reranker = CrossEncoder("BAAI/bge-reranker-base", local_files_only=True)

class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 2


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def semantic_search(where_sql: list[str], sql_params: list[str]) -> List[Tuple[str, Optional[str], str, float]]:
    query = f"""
            SELECT file_path, header_path, content, embedding <-> %s::vector AS distance
            FROM lore_chunks
            {where_sql}
            ORDER BY distance ASC
            LIMIT %s;
        """
    
    try:
        with psycopg.connect(DB_PARAMS) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(query, sql_params)
                results = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database-Error: {e}")

    return results


def fulltext_search(search_term, where_sql, top_k, filter_params) -> List[Tuple[str, Optional[str], str, float]]:
    # TODO dynamische Sprachselektion
    # TODO WHERE Clauseln mitgeben

    if(where_sql):
        conditions = where_sql + "\n AND"
    else:
        conditions = "WHERE"


    query = f"""
        SELECT file_path, header_path, content, ts_rank(content_vector, plainto_tsquery('english', %s)) AS rank
        FROM lore_chunks
        {conditions} content_vector @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s;
    """

    exec_params = [search_term] + filter_params + [search_term, top_k]

    try:
        with psycopg.connect(DB_PARAMS) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(query, exec_params)
                results = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database-Error: {e}")
    
    return results


def reciprocal_rank_fusion_extended(semanticList, indexedList, k=60) -> List[Tuple[float, str, Optional[str], str]]:

    rrf_scores = {}

    def process_list(lst):
        for position, item in enumerate(lst, start=1):
            file_path, header_path, content, original_rank = item

            doc_key = (file_path, header_path, content) # TODO consider unique key from db?

            if doc_key not in rrf_scores:
                rrf_scores[doc_key] = {
                    "score": 0.0,
                    "file_path": file_path,
                    "header_path": header_path,
                    "content": content
                }

            rrf_scores[doc_key]["score"] += 1.0 / (k + position)


    process_list(semanticList)
    process_list(indexedList)

    sorted_fusion = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    return sorted_fusion


def rerank_documents(query: str, retrieved_docs: List[Dict[str, any]], top_k: int = 3) -> list[str]:
    contents = [item["content"] for item in retrieved_docs]

    if not contents:
        return []

    pairs = [[query, doc] for doc in contents]

    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    best_docs = [doc for doc, score in scored_docs[:top_k]]

    return best_docs




@demiurg_api.post("/ask")
def ask_demiurg_endpoint(req: QueryRequest):
    try:
        query_vec = get_embedding(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in the embedding encoding: {e}")

    where_clauses = []
    filter_values_only = []

    if req.filters:
        for key, value in req.filters.items():
            where_clauses.append("metadata->>%s = %s") #TODO complexere Ausdrücke wie NOT müssen dann wohl hier angepasst werden
            filter_values_only.extend([key, str(value)])

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + "AND ".join(where_clauses)


    semantic_params = [query_vec] + filter_values_only + [20]
    # TODO async aufrufe der Suchvorgänge
    results = semantic_search(where_sql, semantic_params)
    # TODO index search
    index_results = fulltext_search(req.question, where_sql, 20, filter_values_only)

    if not results:
        return {
            "answer": "No fitting Lore-Entries found.",
            "sources": []
        }

    #TODO null check index

    ranked_list = reciprocal_rank_fusion_extended(results, index_results)

    reranked_list = rerank_documents(req.question, ranked_list, req.top_k)
    
    context_chunks = []
    sources = []
    for row in reranked_list:
        score = row["score"]
        file_path = row["file_path"]
        header_path = row["header_path"]
        content = row["header_path"]
        
        sources.append({
            "file_path": file_path,
            "header_path": header_path,
            "distance": score
        })
        print(f"Add the following lore to the context: {content}")
        context_chunks.append(content)

    combined_context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are Demiurg-AI, a precise world-building database. "
        "DO NOT INVENT any details, proper nouns, locations, or backstories that are not explicitly mentioned in the context."
        "If the context does not provide the answer, state that directly and do not speculate."
    )

    # system_prompt = (
    #     "You are Demiurg-AI, a precise and strictly isolated world-building database and lore-retrieval engine. "
    #     "You operate EXCLUSIVELY inside the boundaries of the provided context text.\n\n"
        
    #     "CRITICAL CONSTRAINTS:\n"
    #     "1. IGNORE all your pre-trained world knowledge, external lore, books, movies, "
    #     "and general tropes (e.g., Tolkien, Dungeons & Dragons, Middle-earth, etc.). "
    #     "If it is not in the context, it does not exist in your universe.\n"
    #     "ONLY to what the context says. Do not generalize or fill in gaps with external lore.\n"
    #     "2. If the context does not contain the answer, reply ONLY with: "
    #     "\"I do not have enough information in the provided records to answer this.\""
    # )
    user_prompt = f"Context:\n{combined_context}\n\nQuestion: {req.question}"

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in the LLM-Generation: {e}")

    return {
        "answer": response["message"]["content"].strip(),
        "sources": sources
    }

if __name__     == "__main__":
    print("Start Api")
    import uvicorn
    uvicorn.run("demiurg_api:demiurg_api", host="127.0.0.1", port=8000, reload=True)