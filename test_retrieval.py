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
                print(f"\n--- Hits (Distance: {distance:.4f}) ---")
                print("File: {file_path} [{header_path}]")
                print("Metadata: {metadata}")
                print("Content:\n{content.strip()}")

if __name__ == "__main__":
    test_search("Where do dwarves mine mithril?")