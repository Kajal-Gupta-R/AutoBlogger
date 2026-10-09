from langgraph.graph import END, START, StateGraph

from backend.nodes import pick_topic, review, write_draft, write_outline
from backend.state import BlogState


def build_graph():
    graph = StateGraph(BlogState)

    graph.add_node("pick_topic", pick_topic)
    graph.add_node("write_outline", write_outline)
    graph.add_node("write_draft", write_draft)
    graph.add_node("review", review)

    graph.add_edge(START, "pick_topic")
    graph.add_edge("pick_topic", "write_outline")
    graph.add_edge("write_outline", "write_draft")
    graph.add_edge("write_draft", "review")
    graph.add_edge("review", END)

    return graph.compile()
