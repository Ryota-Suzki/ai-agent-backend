import typer
from backend.test.agent.chat import test_chat
from backend.test.agent.embedding import test_embedding
from backend.test.qdrant.ingest import test_ingest
from backend.test.qdrant.init import test_init
from backend.test.qdrant.search import test_search

app = typer.Typer(help="AIエージェントの検証用テストスクリプト")

@app.command()
def init():
    test_init()

@app.command()
def chat():
    test_chat()

@app.command()
def embedding():
    test_embedding()
    
@app.command()
def ingest():
    test_ingest()
    
@app.command()
def search():
    test_search()

if __name__ == "__main__":
    app()