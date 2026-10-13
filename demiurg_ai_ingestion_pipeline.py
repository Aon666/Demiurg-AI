import os
import glob
import hashlib
import frontmatter
import psycopg
from pgvector.psycopg import register_vector
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import ollama

DB_PARAMS = "host=localhost port=5432 dbname=demiurg user=postgres password=postgrespassword"

def setup_database(conn):
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
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]

def process_markdown_file(file_path: str) -> list[dict]:
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
    header_splits = markdown_splitter.split_text(raw_content)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(header_splits)

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
    print("Start Demiurg-AI Ingestion-Pipeline...")
    
    with psycopg.connect(DB_PARAMS) as conn:
        register_vector(conn)
        setup_database(conn)
        
        md_files = glob.glob("data/*.md")
        if not md_files:
            print("No Markdown-Files found in data/...")
            return

        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT file_path, file_hash FROM lore_chunks;")
            db_files = {row[0]: row[1] for row in cur.fetchall()}

            total_new_chunks = 0
            for file_path in md_files:

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        if "NoIngest" in file_content:
                            print(f"[Ignored] NoIngest tag found: {file_path}")
                            continue
                except Exception as e:
                    print(f"[Warning] Could not read file {file_path}: {e}")
                    continue

                current_hash = get_file_hash(file_path)
                
                # Prüfen, ob die Datei bereits in der DB existiert und unverändert ist
                if file_path in db_files and db_files[file_path] == current_hash:
                    print(f"[Skipped] Unchanged: {file_path}")
                    continue
                
                # Falls Datei neu oder geändert: Alte Chunks dieser Datei löschen
                if file_path in db_files:
                    print(f"[Update] File has changed: {file_path}")
                    cur.execute("DELETE FROM lore_chunks WHERE file_path = %s;", (file_path,))
                else:
                    print(f"[New] Process new file: {file_path}")

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
        print(f"\nIngestion finished! {total_new_chunks} new/updated Chunks processed.")

if __name__ == "__main__":
    run_ingestion()