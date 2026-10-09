from backend.graph import build_graph


def main():
    app = build_graph()
    result = app.invoke({})

    print(f"TOPIC: {result['topic']}\n")
    print(result["final"])


if __name__ == "__main__":
    main()
