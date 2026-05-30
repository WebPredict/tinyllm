"""
RAG (Retrieval-Augmented Generation) module for TinyLLM.

Indexes code files and retrieves relevant snippets to augment the model's
context when generating code. This is the first hybrid module — testing
whether retrieval makes a tiny model more useful.

Usage:
    # Build the index
    rag = RAG()
    rag.index_corpus("data/react-ts/corpus.txt")

    # Query
    results = rag.search("toggle button component", top_k=3)

    # Augment a prompt with retrieved context
    augmented = rag.augment_prompt("Create a toggle button", top_k=3)
"""

import os
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


class RAG:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "react_ts"):
        if persist_dir is None:
            persist_dir = str(Path(__file__).parent.parent / "data" / "rag_index")

        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.persist_dir = persist_dir

    def index_corpus(self, corpus_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Parse corpus into chunks and index them."""
        corpus_path = Path(corpus_path)
        print(f"Indexing: {corpus_path}")

        # Check if already indexed
        existing = self.collection.count()
        if existing > 0:
            print(f"  Collection already has {existing:,} chunks. Use reindex=True to rebuild.")
            return existing

        text = corpus_path.read_text(encoding="utf-8")

        # Parse into files
        files = []
        current_path = None
        current_lines = []

        for line in text.split("\n"):
            if line.startswith("// FILE: "):
                if current_path and current_lines:
                    files.append({"path": current_path, "content": "\n".join(current_lines)})
                current_path = line[len("// FILE: "):]
                current_lines = []
            else:
                current_lines.append(line)

        if current_path and current_lines:
            files.append({"path": current_path, "content": "\n".join(current_lines)})

        print(f"  Parsed {len(files):,} files")

        # Chunk files
        chunks = []
        for f in files:
            file_chunks = self._chunk_text(f["content"], chunk_size, chunk_overlap)
            for i, chunk in enumerate(file_chunks):
                if len(chunk.strip()) < 30:
                    continue
                chunk_id = hashlib.md5(f"{f['path']}:{i}:{chunk[:50]}".encode()).hexdigest()
                chunks.append({
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        "file": f["path"],
                        "chunk_index": i,
                        "char_count": len(chunk),
                    },
                })

        print(f"  Created {len(chunks):,} chunks (avg {sum(c['metadata']['char_count'] for c in chunks) // len(chunks)} chars)")

        # Index in batches (ChromaDB has batch size limits)
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            if (i + batch_size) % 5000 == 0 or i + batch_size >= len(chunks):
                print(f"  Indexed {min(i + batch_size, len(chunks)):,}/{len(chunks):,} chunks")

        print(f"  Done! {self.collection.count():,} chunks indexed.")
        return self.collection.count()

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> list:
        """Split text into overlapping chunks, respecting line boundaries."""
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if current_size + line_len > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                # Keep overlap lines
                overlap_lines = []
                overlap_size = 0
                for prev_line in reversed(current_chunk):
                    if overlap_size + len(prev_line) + 1 > overlap:
                        break
                    overlap_lines.insert(0, prev_line)
                    overlap_size += len(prev_line) + 1
                current_chunk = overlap_lines
                current_size = overlap_size

            current_chunk.append(line)
            current_size += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def search(self, query: str, top_k: int = 5) -> list:
        """Search for relevant code chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        return hits

    def augment_prompt(self, query: str, top_k: int = 3, max_context_chars: int = 1500) -> str:
        """Build an augmented prompt with retrieved context."""
        hits = self.search(query, top_k=top_k)

        context_parts = []
        total_chars = 0
        for hit in hits:
            text = hit["text"].strip()
            if total_chars + len(text) > max_context_chars:
                break
            context_parts.append(f"// From: {hit['metadata']['file']}\n{text}")
            total_chars += len(text)

        if not context_parts:
            return query

        context = "\n\n".join(context_parts)
        return f"// Related code:\n{context}\n\n// Task:\n{query}\n"

    def stats(self) -> dict:
        """Get index statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "persist_dir": self.persist_dir,
        }
