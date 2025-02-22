from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from utils.vectorstore import (
    get_llm,
    prebuild_sample_vector_store,
)


# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_model: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    llm = get_llm(state["chat_model"])
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph():
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph


vector_store = prebuild_sample_vector_store()
graph = build_graph()


if __name__ == "__main__":
    response = graph.invoke(
        {
            "question": "What is Task Decomposition?",
            "chat_model": "gpt-4o-mini",
        }
    )
    print(response["answer"])
