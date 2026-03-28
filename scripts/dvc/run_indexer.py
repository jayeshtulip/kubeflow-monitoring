"""
CLI wrapper for the Qdrant indexer — used by DVC pipeline stages.

Usage:
  python scripts/dvc/run_indexer.py --mode preprocess --input-dir data/raw --output-dir data/processed
  python scripts/dvc/run_indexer.py --mode embed --input-dir data/processed --output-dir data/embeddings
  python scripts/dvc/run_indexer.py --mode index --input-dir data/embeddings
"""
import argparse, os, json, pathlib, sys, time

def run_preprocess(input_dir: str, output_dir: str, chunk_size: int, overlap: int):
    """Chunk all .txt files in input_dir -> output_dir as JSON chunk files."""
    from src.storage.qdrant.indexer import chunk_text

    input_path  = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Domain mapping based on subdirectory name
    domain_map = {"tech": "tech_docs", "hr": "hr_policies", "org": "org_info"}

    total_chunks = 0
    processed_files = 0
    metrics = {}

    for txt_file in input_path.rglob("*.txt"):
        domain = txt_file.parent.name  # tech / hr / org
        collection = domain_map.get(domain, "tech_docs")
        doc_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")

        chunks = chunk_text(
            text=text,
            doc_id=doc_id,
            source=str(txt_file),
            collection=collection,
            chunk_size=chunk_size,
            overlap=overlap,
            extra_metadata={"domain": domain},
        )

        # Save chunks as JSON
        out_file = output_path / f"{domain}_{doc_id}_chunks.json"
        with open(out_file, "w") as f:
            json.dump([{
                "text": c.text, "chunk_index": c.chunk_index,
                "doc_id": c.doc_id, "source": c.source,
                "collection": c.collection,
                "extra_metadata": c.extra_metadata
            } for c in chunks], f, indent=2)

        total_chunks += len(chunks)
        processed_files += 1
        print(f"  {txt_file.name} -> {len(chunks)} chunks -> {out_file.name}")

    metrics = {"files": processed_files, "total_chunks": total_chunks}
    with open(output_path / "preprocess_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nPreprocess complete: {processed_files} files, {total_chunks} chunks")
    return metrics


def run_embed(input_dir: str, output_dir: str, model_name: str):
    """Load chunks from input_dir, generate embeddings, save to output_dir."""
    from src.storage.qdrant.indexer import embed_chunks, Chunk

    input_path  = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_embedded = 0

    for chunk_file in input_path.glob("*_chunks.json"):
        with open(chunk_file) as f:
            raw_chunks = json.load(f)

        chunks = [Chunk(
            text=c["text"], chunk_index=c["chunk_index"],
            doc_id=c["doc_id"], source=c["source"],
            collection=c["collection"],
            start_word=0, end_word=0,
            extra_metadata=c.get("extra_metadata", {})
        ) for c in raw_chunks]

        chunk_embeddings = embed_chunks(chunks, model_name=model_name)

        # Save embeddings as JSON
        out_file = output_path / chunk_file.name.replace("_chunks.json", "_embeddings.json")
        with open(out_file, "w") as f:
            json.dump([{
                "chunk": {
                    "text": ch.text, "chunk_index": ch.chunk_index,
                    "doc_id": ch.doc_id, "source": ch.source,
                    "collection": ch.collection,
                    "extra_metadata": ch.extra_metadata
                },
                "vector": emb
            } for ch, emb in chunk_embeddings], f)

        total_embedded += len(chunks)
        print(f"  {chunk_file.name} -> {len(chunks)} embeddings -> {out_file.name}")

    metrics = {"total_embedded": total_embedded}
    with open(output_path / "embed_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEmbed complete: {total_embedded} vectors generated")
    return metrics


def run_index(input_dir: str, qdrant_host: str, qdrant_port: int):
    """Load embeddings from input_dir, upsert to Qdrant."""
    from src.storage.qdrant.indexer import upsert_to_qdrant, Chunk
    from qdrant_client import QdrantClient

    input_path = pathlib.Path(input_dir)
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    total_upserted = 0
    all_metrics = {}

    for emb_file in input_path.glob("*_embeddings.json"):
        with open(emb_file) as f:
            data = json.load(f)

        chunk_embeddings = [(
            Chunk(
                text=d["chunk"]["text"],
                chunk_index=d["chunk"]["chunk_index"],
                doc_id=d["chunk"]["doc_id"],
                source=d["chunk"]["source"],
                collection=d["chunk"]["collection"],
                start_word=0, end_word=0,
                extra_metadata=d["chunk"].get("extra_metadata", {})
            ),
            d["vector"]
        ) for d in data]

        metrics = upsert_to_qdrant(chunk_embeddings, client)
        all_metrics[emb_file.name] = metrics
        total_upserted += metrics["chunks_upserted"]
        print(f"  {emb_file.name} -> {metrics['chunks_upserted']} points upserted to Qdrant:{metrics['collection']}")

    metrics_file = input_path / "index_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({"total_upserted": total_upserted, "by_file": all_metrics}, f, indent=2)
    print(f"\nIndex complete: {total_upserted} total vectors upserted to Qdrant")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Qdrant DVC pipeline indexer")
    parser.add_argument("--mode", required=True, choices=["preprocess", "embed", "index"])
    parser.add_argument("--input-dir",  default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--overlap",    type=int, default=10)
    parser.add_argument("--model",      default="all-MiniLM-L6-v2")
    parser.add_argument("--qdrant-host", default=os.environ.get("QDRANT_HOST", "qdrant-service.llm-platform-prod.svc.cluster.local"))
    parser.add_argument("--qdrant-port", type=int, default=6333)
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    t0 = time.time()

    if args.mode == "preprocess":
        run_preprocess(args.input_dir, args.output_dir, args.chunk_size, args.overlap)
    elif args.mode == "embed":
        run_embed(args.input_dir, args.output_dir, args.model)
    elif args.mode == "index":
        run_index(args.input_dir, args.qdrant_host, args.qdrant_port)

    print(f"\nTotal time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
