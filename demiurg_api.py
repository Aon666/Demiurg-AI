from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg
from pgvector.psycopg import register_vector
import ollama
from typing import Optional, Dict, Any

demiurg_api = FastAPI(
    title="Demiurg-AI API",
    description="Local-First RAG Assistant for TTRPG Worldbuilding & Lore Management",
    version="1.0"
)

DB_PARAMS = "host=localhost port=5432 dbname=demiurg user=postgres password=postgrespassword"

class QueryRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 2

def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

@demiurg_api.post("/ask")
def ask_demiurg_endpoint(req: QueryRequest):
    try:
        query_vec = get_embedding(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in the embedding encoding: {e}")

    where_clauses = []
    sql_params = [query_vec]

    if req.filters:
        for key, value in req.filters.items():
            where_clauses.append("metadata->>%s = %s") #TODO complexere Ausdrücke wie NOT müssen dann wohl hier angepasst werden
            sql_params.extend([key, str(value)])

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + "AND ".join(where_clauses)

    sql_params.append(req.top_k)

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

    if not results:
        return {
            "answer": "No fitting Lore-Entries found.",
            "sources": []
        }

    
    context_chunks = []
    sources = []
    for row in results:
        file_path, header_path, content, distance = row
        sources.append({
            "file_path": file_path,
            "header_path": header_path,
            "distance": float(distance)
        })
        context_chunks.append(content)

    combined_context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are Demiurg-AI, a precise world-building database. "
        "DO NOT INVENT any details, proper nouns, locations, or backstories that are not explicitly mentioned in the context."
        "If the context does not provide the answer, state that directly and do not speculate."
    )
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
    import uvicorn
    uvicorn.run("demiurg_api:demiurg_api", host="127.0.0.1", port=8000, reload=True)