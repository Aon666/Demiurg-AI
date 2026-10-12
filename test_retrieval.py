import psycopg
from pgvector.psycopg import register_vector
import ollama

DB_PARAMS = "host=localhost port=5432 dbname=worldsmith user=postgres password=postgrespassword"

def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def test_search(query: str):
    print(query)
    query_vec = get_embedding(query)
    
    with psycopg.connect(DB_PARAMS) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT file_path, header_path, metadata, content, embedding <-> %s::vector AS distance 
                FROM lore_chunks 
                ORDER BY distance ASC 
                LIMIT 2;
                """,
                (query_vec,)
            )
            results = cur.fetchall()
            
            for row in results:
                file_path, header_path, metadata, content, distance = row
                print(f"\n--- Treffer (Distanz: {distance:.4f}) ---")
                print("Datei: {file_path} [{header_path}]")
                print("Metadaten: {metadata}")
                print("Inhalt:\n{content.strip()}")

if __name__ == "__main__":
    test_search("Wo bauen die Zwerge Erze ab?")