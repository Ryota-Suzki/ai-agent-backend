
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from backend.src.agent.ai_agent import get_ollama_embeddings
from backend.src.qdrant.qdrant import upsert_code_chunks
from backend.src.config import settings


def test_ingest():
    print("\n=== データインジェストのテストを開始 ===")
    
    # 1. テスト用の擬似コード（リポジトリAに含まれている想定のファイル）
    dummy_code = """
    // userController.ts
    export const getUser = async (id: string) => {
        const user = await db.users.findUnique({ where: { id } });
        if (!user) {
            throw new Error("User not found");
        }
        return user;
    };

    export const createUser = async (data: any) => {
        return await db.users.create({ data });
    };
    """
    
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.TS,
        chunk_size=150, 
        chunk_overlap=20
    )
    
    chunks = splitter.split_text(dummy_code)
    print(f"✂️ コードを {len(chunks)} 個のチャンクに分割")
    for i, c in enumerate(chunks):
        print(f"--- チャンク [{i}] ---\n{c.strip()}")
        
    target_repo_id = settings.repository_id
    
    embeddings = get_ollama_embeddings()
    
    upsert_code_chunks(
        chunks=chunks,
        embeddings_model=embeddings,
        repository_id=target_repo_id
    )
    
    print("=== データインジェストのテストを終了 ===\n")