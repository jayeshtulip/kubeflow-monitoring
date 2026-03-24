"""
CLI entrypoint for DVC pipeline stages.
Called by dvc.yaml as: python -m src.storage.qdrant.indexer --mode <stage>

Modes:
  preprocess  --input-dir data/raw --output-dir data/processed
  embed       --input-dir data/processed --output-dir data/embeddings
  index       --input-dir data/embeddings
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def run_preprocess(input_dir: str, output_dir: str, params: dict) -> dict:
    """Chunk raw text files into overlapping segments."""
    from src.storage.qdrant.indexer import chunk_text

    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chunk_size = params.get("rag", {}).get("chunk_size", 150)
    overlap    = params.get("rag", {}).get("overlap", 30)

    total_docs   = 0
    total_chunks = 0
    t_start      = time.perf_counter()

    for txt_file in sorted(in_path.glob("**/*.txt")):
        doc_text = txt_file.read_text(encoding="utf-8")
        doc_id   = txt_file.stem
        chunks   = chunk_text(
            text=doc_text,
            doc_id=doc_id,
            source=txt_file.parent.name,
            collection="tech_docs",
            chunk_size=chunk_size,
            overlap=overlap,
        )
        out_file = out_path / f"{doc_id}.json"
        out_file.write_text(
            json.dumps([c.__dict__ for c in chunks], indent=2, default=str),
            encoding="utf-8",
        )
        total_docs   += 1
        total_chunks += len(chunks)
        print(f"  Chunked {doc_id}: {len(chunks)} chunks")

    metrics = {
        "total_docs":   total_docs,
        "total_chunks": total_chunks,
        "chunk_size":   chunk_size,
        "overlap":      overlap,
        "elapsed_s":    round(time.perf_counter() - t_start, 2),
    }
    (out_path / "preprocess_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Preprocess done: {total_docs} docs, {total_chunks} chunks")
    return metrics


def run_embed(input_dir: str, output_dir: str, params: dict) -> dict:
    """Generate sentence-transformer embeddings for all chunks."""
    from sentence_transformers import SentenceTransformer

    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_name = params.get("rag", {}).get("embedding_model", "all-MiniLM-L6-v2")
    model      = SentenceTransformer(model_name)

    total_vectors = 0
    t_start       = time.perf_counter()

    for chunk_file in sorted(in_path.glob("*.json")):
        if chunk_file.name.endswith("_metrics.json"):
            continue
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        texts  = [c["text"] for c in chunks]
        if not texts:
            continue
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for chunk, vec in zip(chunks, vectors):
            chunk["vector"] = vec.tolist()
        out_file = out_path / chunk_file.name
        out_file.write_text(
            json.dumps(chunks, indent=2, default=str), encoding="utf-8"
        )
        total_vectors += len(chunks)
        print(f"  Embedded {chunk_file.stem}: {len(chunks)} vectors")

    metrics = {
        "total_vectors": total_vectors,
        "model":         model_name,
        "dims":          params.get("rag", {}).get("embedding_dims", 384),
        "elapsed_s":     round(time.perf_counter() - t_start, 2),
    }
    (out_path / "embed_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Embed done: {total_vectors} vectors from {model_name}")
    return metrics


def run_index(input_dir: str, params: dict) -> dict:
    """Upsert embeddings into Qdrant collections."""
    from src.storage.qdrant.indexer import upsert_to_qdrant
    from pipelines.components.shared.base import EnvConfig, get_qdrant_client, ensure_qdrant_collection

    in_path = Path(input_dir)
    cfg     = EnvConfig()
    client  = get_qdrant_client(cfg)
    dims    = params.get("rag", {}).get("embedding_dims", 384)

    total_upserted = 0
    t_start        = time.perf_counter()

    for embed_file in sorted(in_path.glob("*.json")):
        if embed_file.name.endswith("_metrics.json"):
            continue
        chunks = json.loads(embed_file.read_text(encoding="utf-8"))
        if not chunks:
            continue
        collection = chunks[0].get("collection", "tech_docs")
        ensure_qdrant_collection(client, collection, dims=dims)
        upsert_to_qdrant(chunks, client)
        total_upserted += len(chunks)
        print(f"  Indexed {embed_file.stem}: {len(chunks)} vectors -> {collection}")

    metrics = {
        "total_upserted": total_upserted,
        "qdrant_host":    cfg.qdrant_host,
        "elapsed_s":      round(time.perf_counter() - t_start, 2),
    }
    (in_path / "index_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Index done: {total_upserted} vectors upserted to Qdrant")
    return metrics


def load_params(params_file: str) -> dict:
    if not params_file or not os.path.exists(params_file):
        return {}
    import yaml
    with open(params_file) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="DVC pipeline stage runner")
    parser.add_argument("--mode",       required=True, choices=["preprocess", "embed", "index"])
    parser.add_argument("--input-dir",  default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--params",     default="mlops/dvc/params/rag_params.yaml")
    args = parser.parse_args()

    params = load_params(args.params)
    print(f"Running stage: {args.mode}")

    if args.mode == "preprocess":
        run_preprocess(args.input_dir, args.output_dir, params)
    elif args.mode == "embed":
        run_embed(args.input_dir, args.output_dir, params)
    elif args.mode == "index":
        run_index(args.input_dir, params)


if __name__ == "__main__":
    main()
