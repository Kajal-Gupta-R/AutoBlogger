from backend.graph import build_graph


def main():
    app = build_graph()
    result = app.invoke({})
    print(f"Saved: frontend/articles/{result['filename']}")
    print(f"Topic: {result['topic']}")


if __name__ == "__main__":
    main()
