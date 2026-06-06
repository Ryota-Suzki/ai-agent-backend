from datetime import datetime
from typing import TypedDict

from backend.src.config import settings
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    input: str
    search_results: str
    output: str
    messages: list


async def process_input(state: State) -> State:
    raw_knowledge = state.get("search_results", "")
    if (
        raw_knowledge
        and raw_knowledge != "該当するコードナレッジは見つかりませんでした。"
    ):
        formatted_knowledge = (
            f"【ローカルナレッジ（ソースコード・ドキュメント）】\n{raw_knowledge}"
        )
    else:
        formatted_knowledge = ""

    return {
        "messages": [f"ユーザー入力: {state['input']}"],
        "input": state["input"],
        "search_results": formatted_knowledge,
        "output": "",
    }


async def generate_response(state: State) -> State:
    llm = OllamaLLM(
        model=settings.llm_model, base_url=settings.ollama_port, temperature=0.7
    )
    current_time = datetime.now().strftime("%Y年%m月%d日 %H時%M分")

    prompt = f"""あなたは提供された情報ソースに基づいて的確に回答する優秀なエンジニアアシスタントです。
あなたの一般的な知識や推測で回答を作成せず、必ず以下の【提供された情報ソース】の内容を最優先して回答してください。

現在のシステム日時は {current_time} です。

【ユーザーの質問】
{state["input"]}

【提供された情報ソース】
{state.get("search_results", "情報なし")}

【回答ルール】
1. 提供された情報ソースに該当情報がある場合は、その関数名やロジック、ドキュメントの仕様を正確に引用して回答してください。
2. 提供された情報ソースのどこにも質問に対する直接的な答えが見つからない場合は、勝手に推測や幻覚で答えを作らず、「手元のナレッジからは該当する情報が見つかりませんでした」と正直に回答してください。
"""

    response = await llm.ainvoke(prompt)
    return {
        "messages": state["messages"] + ["回答生成完了"],
        "input": state["input"],
        "search_results": state["search_results"]
        if state.get("search_results")
        else "（ナレッジなし）",
        "output": response,
    }


def create_graph():
    """LangGraphを構築（外部検索なしの完全クローズドフロー）"""
    graph_builder = StateGraph(State)

    graph_builder.add_node("process_input", process_input)
    graph_builder.add_node("generate_response", generate_response)

    graph_builder.add_edge(START, "process_input")
    graph_builder.add_edge("process_input", "generate_response")
    graph_builder.add_edge("generate_response", END)

    return graph_builder.compile()


def get_ollama_embeddings():
    """Ollamaの埋め込みモデル（bge-m3）を取得する共通部品"""
    return OllamaEmbeddings(
        model=settings.embeddings_model, base_url=settings.ollama_port
    )


async def stream_agent_response(user_input: str, raw_knowledge: str = ""):
    """LangGraphを実行し、LLMのトークンをリアルタイムでストリーミングするジェネレーター"""
    graph = create_graph()
    initial_state = {
        "input": user_input,
        "search_results": raw_knowledge,
        "messages": [],
    }

    async for event in graph.astream_events(initial_state, version="v2"):
        kind = event["event"]

        if kind == "on_llm_stream":
            chunk = event["data"]["chunk"]
            text = chunk.content if hasattr(chunk, "content") else chunk

            yield text
