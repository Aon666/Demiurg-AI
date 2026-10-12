from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg
from pgvector.psycopg import register_vector
import ollama
from typing import Optional, Dict, Any

app = FastAPI(
    title="Demiurg-AI API",
    description="Lokales RAG-Backend für TTRPG-Worldbuilding und Lore-Management",
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

@app.post("/ask")
def ask_demiurg_endpoint(req: QueryRequest):
    try:
        query_vec = get_embedding(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Embedding-Generierung: {e}")

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
        raise HTTPException(status_code=500, detail=f"Datenbankfehler: {e}")

    if not results:
        return {
            "answer": "Keine passende Lore-Einträge gefunden.",
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
        "Du bist Demiurg-AI, eine exakte Worldbuilding-Datenbank. "
        "Beantworte die Frage AUSSCHLIESSLICH und wortgetreu basierend auf dem bereitgestellten Kontext. "
        "ERFINDE KEINE Details, Eigennamen, Orte oder Hintergrundgeschichten hinzu, die nicht explizit im Kontext stehen. "
        "Wenn der Kontext die Antwort nicht hergibt, sage das direkt und spekuliere nicht."
    )
    user_prompt = f"Kontext:\n{combined_context}\n\nFrage: {req.question}"

    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der LLM-Generierung: {e}")

    return {
        "answer": response["message"]["content"].strip(),
        "sources": sources
    }

if __name__     == "__main__":
    import uvicorn
    uvicorn.run("phase4_api:phase4_api" if __name__ == "__main__" else "phase4_api", host="127.0.0.1", port=8000, reload=True)