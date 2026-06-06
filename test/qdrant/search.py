from backend.src.agent.ai_agent import get_ollama_embeddings
from backend.src.config import settings
from backend.src.qdrant.qdrant import search_code_knowledge


def test_search():
    print("\n=== ベクトル検索のテストを開始 ===")

    # 1. テスト用のクエリと対象リポジトリ
    test_query = "ユーザーを作成する関数はある？"
    target_repo_id = settings.repository_id

    # 2. Ollama(bge-m3)を取得
    embeddings = get_ollama_embeddings()

    results = search_code_knowledge(
        query=test_query,
        embeddings_model=embeddings,
        repository_id=target_repo_id,
        top_k=2,
    )

    print(f"\n🎯 検索結果: {len(results)} 件ヒット")
    print("-" * 50)
    for i, hit in enumerate(results):
        score = hit.score
        content = hit.payload.get("content", "N/A").strip()
        print(f"[{i + 1}] 類似度スコア: {score:.4f}")
        print(f"📄 コード断片:\n{content}")
        print("-" * 50)

    print("=== ベクトル検索のテストを終了 ===\n")
