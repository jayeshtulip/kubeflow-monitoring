"""L2 Component tests — Qdrant vector database."""
import pytest
import uuid
from qdrant_client.http.models import PointStruct, VectorParams, Distance

TEST_COLLECTION = "l2_test_collection"


@pytest.mark.l2
@pytest.mark.timeout(15)
def test_qdrant_health(qdrant_client):
    """Assert Qdrant responds to get_collections()."""
    cols = qdrant_client.get_collections()
    assert cols is not None


@pytest.mark.l2
@pytest.mark.timeout(30)
def test_qdrant_upsert_and_query(qdrant_client):
    """Upsert 10 test vectors, query top-5, assert cosine >= 0.6."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create temp collection
    if TEST_COLLECTION in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.delete_collection(TEST_COLLECTION)
    qdrant_client.create_collection(
        TEST_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Upsert test documents
    docs = [
        "The EKS payment service experienced pod evictions due to memory pressure",
        "Pod evictions at 14:20 caused orphaned RDS connections",
        "Connection pool exhaustion led to payment service timeouts",
        "CronJob daily-report-generator started at 14:15 consuming excess memory",
        "The root cause was the CronJob triggering memory pressure on all nodes",
    ]
    vectors = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v.tolist(),
                    payload={"text": t, "doc_id": f"test-{i}"})
        for i, (t, v) in enumerate(zip(docs, vectors))
    ]
    qdrant_client.upsert(TEST_COLLECTION, points)

    # Query
    query = "payment service timeout root cause"
    qvec = model.encode(query, normalize_embeddings=True).tolist()
    hits = qdrant_client.search(TEST_COLLECTION, query_vector=qvec, limit=3)

    assert len(hits) >= 3, f"Expected >= 3 results, got {len(hits)}"
    assert hits[0].score >= 0.6, f"Top score {hits[0].score:.3f} < 0.7"

    # Cleanup
    qdrant_client.delete_collection(TEST_COLLECTION)


@pytest.mark.l2
@pytest.mark.timeout(15)
@pytest.mark.live
def test_production_collections_exist(qdrant_client):
    """Assert tech_docs, hr_policies, org_info collections exist."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    for col in ["tech_docs", "hr_policies", "org_info"]:
        assert col in existing, f"Collection {col!r} not found in Qdrant"

