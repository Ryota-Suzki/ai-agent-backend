
from backend.src.agent.ai_agent import State, create_graph


def test_chat():
    print("🤖 AIエージェント CLIモード (終了するには 'q' を入力)")
    print("-" * 50)
    
    # グラフの初期化
    graph = create_graph()

    while True:
        # ターミナルから入力を受け付ける
        query = input("\n質問: ").strip()
        
        if query.lower() in ['q', 'exit', 'quit']:
            print("終了")
            break
            
        if not query:
            continue

        print("\n実行中...\n")

        input_data: State = {
            "input": query,
            "search_query": "",
            "search_results": "",
            "output": "",
            "messages": []
        }

        try:
            # グラフを実行
            result = graph.invoke(input_data)
            
            # 回答の表示
            print("=" * 20 + " 回答 " + "=" * 20)
            print(result["output"])
            print("=" * 46)
                
        except Exception as e:
            print(f"エラーが発生しました: {e}")