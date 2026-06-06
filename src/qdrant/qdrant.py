import uuid

from backend.src.agent.ai_agent import get_ollama_embeddings
from backend.src.config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


def get_qdrant_client():
    """Qdrantクライアントを取得する共通部品"""
    return QdrantClient(host="localhost", port=settings.qdrant_port)


def init_database():
    """コレクションを初期化する部品"""
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if exists:
        print(f"⚠️ コレクション '{collection_name}' は既に存在します。")
    else:
        print(f"🚀 コレクション '{collection_name}' を作成します...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"✅ コレクション '{collection_name}' の作成に成功しました！")


def format_vector(message: str):
    embeddings = get_ollama_embeddings()
    return embeddings.embed_query(message)


def upsert_code_chunks(chunks: list[str], embeddings_model, repository_id: str):
    """
    分割されたチャンク（テキストリスト）をベクトル化し、Qdrantに保存する部品
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    vectors = embeddings_model.embed_documents(chunks)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "repository_id": repository_id,
                    "content": chunk,
                    "chunk_index": i,
                },
            )
        )

    client.upsert(collection_name=collection_name, wait=True, points=points)


def search_code_knowledge(
    query: str, embeddings_model, repository_id: str, top_k: int = 3
):
    """
    クエリをベクトル化し、特定のリポジトリIDに絞り込んでQdrantから検索する部品
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    query_vector = embeddings_model.embed_query(query)

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="repository_id", match=MatchValue(value=repository_id)
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return response.points
