from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4o-mini")

from typing import Annotated

# from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def process_hil(values):
    if "messages" in values and len(values["messages"]) > 0:
        if "tool_calls" in values["messages"][-1].additional_kwargs:
            tool_calls = values["messages"][-1].additional_kwargs["tool_calls"]
            if tool_calls[0]["function"]["name"] == "human_assistance":
                prompt = json.loads(tool_calls[0]["function"]["arguments"])["query"]
                resume_key = "data"
                return prompt, resume_key
    return None, None


def process_interrupt(snapshot):
    # TODO: assume only one task, one interrupt
    interrupt = {}
    if snapshot.next:
        for task in snapshot.tasks:
            for interrupt_ in task.interrupts:
                # NOTE: interrupt_.value can be a dictionary, e.g. for options
                interrupt["prompt"] = interrupt_.value
    return interrupt


def test_hil():
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(config)
    print(snapshot.next)

    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


def test_hil_interactive():
    config = {"configurable": {"thread_id": "1"}}

    interrupt = {}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if interrupt:
            # TODO: handle resume as a dict and interrupt.value as a dict
            human_command = Command(resume={"data": user_input})
            response = graph.invoke(human_command, config)
        else:
            response = graph.invoke(
                {"messages": [{"role": "user", "content": user_input}]}, config
            )

        snapshot = graph.get_state(config)
        interrupt = process_interrupt(snapshot)
        response["messages"][-1].pretty_print()

        if interrupt:
            print(f"Assistant (prompt): {interrupt['prompt']}")


if __name__ == "__main__":
    test_hil_interactive()
