import os
import glob
import hashlib
import frontmatter
import psycopg
from pgvector.psycopg import register_vector
from langchain_text_splitters import MarkdownHeaderTextSplitter
import ollama

DB_PARAMS = "host=localhost port=5432 dbname=worldsmith user=postgres password=postgrespassword"

def setup_phase2_database(conn):
    """Aktualisiert das Schema für Metadaten, JSONB und Datei-Hashes."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS lore_chunks (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                header_path TEXT,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector(768)
            );
        """)
    conn.commit()

def get_file_hash(file_path: str) -> str:
    """Berechnet einen MD5-Hash des Dateiinhaltes zur Änderungserkennung."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_embedding(text: str) -> list[float]:
    """Generiert ein Embedding-Vektor für einen Text über Ollama."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def process_markdown_file(file_path: str) -> list[dict]:
    """Liest eine Markdown-Datei, parst YAML-Frontmatter und chunkt nach Headern."""
    post = frontmatter.load(file_path)
    yaml_metadata = post.metadata
    raw_content = post.content

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    splits = markdown_splitter.split_text(raw_content)

    file_hash = get_file_hash(file_path)
    chunks = []
    for split in splits:
        header_path_parts = [val for key, val in split.metadata.items()]
        header_path = " > ".join(header_path_parts) if header_path_parts else "General"

        chunks.append({
            "file_path": file_path,
            "file_hash": file_hash,
            "header_path": header_path,
            "content": split.page_content,
            "metadata": yaml_metadata
        })
    
    return chunks

def run_ingestion():
    print("Starte Demiurg-AI Ingestion-Pipeline...")
    
    with psycopg.connect(DB_PARAMS) as conn:
        register_vector(conn)
        setup_phase2_database(conn)
        
        md_files = glob.glob("data/*.md")
        if not md_files:
            print("Keine Markdown-Dateien im Ordner 'data/' gefunden!")
            return

        with conn.cursor() as cur:
            # Aktuell gespeicherte Hashes aus der DB auslesen
            cur.execute("SELECT DISTINCT file_path, file_hash FROM lore_chunks;")
            db_files = {row[0]: row[1] for row in cur.fetchall()}

            total_new_chunks = 0
            for file_path in md_files:
                current_hash = get_file_hash(file_path)
                
                # Prüfen, ob die Datei bereits in der DB existiert und unverändert ist
                if file_path in db_files and db_files[file_path] == current_hash:
                    print(f"[Übersprungen] Unverändert: {file_path}")
                    continue
                
                # Falls Datei neu oder geändert: Alte Chunks dieser Datei löschen
                if file_path in db_files:
                    print(f"[Aktualisierung] Datei hat sich geändert: {file_path}")
                    cur.execute("DELETE FROM lore_chunks WHERE file_path = %s;", (file_path,))
                else:
                    print(f"[Neu] Verarbeite neue Datei: {file_path}")

                chunks = process_markdown_file(file_path)
                for chunk in chunks:
                    vec = get_embedding(chunk["content"])
                    cur.execute(
                        """
                        INSERT INTO lore_chunks (file_path, file_hash, header_path, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            chunk["file_path"],
                            chunk["file_hash"],
                            chunk["header_path"],
                            chunk["content"],
                            psycopg.types.json.Jsonb(chunk["metadata"]),
                            vec
                        )
                    )
                    total_new_chunks += 1

        conn.commit()
        print(f"\nIngestion abgeschlossen! {total_new_chunks} neue/aktualisierte Chunks verarbeitet.")

if __name__ == "__main__":
    run_ingestion()