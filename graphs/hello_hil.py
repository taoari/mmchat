from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
# from IPython.display import Image, display


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state):
    print("---Step 1---")
    pass


def human_feedback(state):
    print("---human_feedback---")
    feedback = interrupt("Please provide feedback:")
    return {"user_feedback": feedback}


def step_3(state):
    print("---Step 3---")
    pass


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory)

# View
# display(Image(graph.get_graph().draw_mermaid_png()))


def test_hil():
    # Input
    initial_input = {"input": "hello world"}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode=["updates", "values"]):
        print(event)
        print("\n")

    # Continue the graph execution
    for event in graph.stream(
        Command(resume="go to step 3!"), thread, stream_mode=["updates", "values"]
    ):
        print(event)
        print("\n")


def process_interrupt(snapshot):
    # TODO: assume only one task, one interrupt
    interrupt = {}
    if snapshot.next:
        for task in snapshot.tasks:
            for interrupt_ in task.interrupts:
                interrupt["prompt"] = interrupt_.value
    return interrupt


def test_hil_interactive():
    config = {"configurable": {"thread_id": "1"}}

    interrupt = {}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if interrupt:
            human_command = Command(resume=user_input)
            response = graph.invoke(human_command, config)
        else:
            response = graph.invoke({"input": user_input}, config)

        snapshot = graph.get_state(config)
        interrupt = process_interrupt(snapshot)
        print(response)

        if interrupt:
            print(f"Assistant (prompt): {interrupt['prompt']}")


if __name__ == "__main__":
    test_hil_interactive()
