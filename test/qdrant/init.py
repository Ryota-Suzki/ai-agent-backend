from backend.src.qdrant.qdrant import init_database


def test_init():
    """Qdrantの初期化とコレクション作成のテスト"""
    print("\n=== Qdrantの初期化テストを開始 ===")
    init_database()
    print("=== Qdrantの初期化テストを終了 ===\n")
