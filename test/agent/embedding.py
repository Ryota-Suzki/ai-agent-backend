
from backend.src.agent.ai_agent import get_ollama_embeddings


def test_embedding():
    print("\n=== Ollama(bge-m3) の埋め込みテストを開始 ===")
    
    # 共通部品からモデルを呼び出し
    embeddings = get_ollama_embeddings()
    
    test_text = "export const useChat = () => {}"
    print(f"📄 テストするテキスト: '{test_text}'")
    
    # テキストをベクトルに変換
    vector = embeddings.embed_query(test_text)
    
    print(f"📊 ベクトルの次元数: {len(vector)} ")
    print(f"🔢 ベクトルの冒頭5要素のプレビュー: {vector[:5]}")
    print("=== Ollama(bge-m3) の埋め込みテストを終了 ===\n")