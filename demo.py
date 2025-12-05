import os
from retrieval_engine import USEEmbedder, build_index


def main():
    # Load knowledge base
    script_dir = os.path.dirname(__file__)
    kb_path = os.path.join(script_dir, "knowledge_base.txt")

    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            document = f.read()
    except FileNotFoundError:
        print(f"Error: knowledge_base.txt not found at {kb_path}")
        print(
            "Please make sure the file exists and is in the same directory as demo.py."
        )
        return

    # Initialize embedder and build index
    embedder = USEEmbedder()
    retrieval_index = build_index(document, embedder)
    print(f"Indexed {len(retrieval_index.sentences)} sentences.")

    # Sample queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain Bayes' rule.",
        "What is overfitting?",
        "How do GANs operate?",
    ]

    print("\n--- Running Sample Queries ---")
    for i, query in enumerate(queries):
        print(f"\nQuery {i + 1}: '{query}'")
        results = retrieval_index.search(query, embedder, top_k=3)  # Retrieve top 3

        for j, result in enumerate(results):
            print(f"  Result {j + 1} (Similarity: {result['similarity']:.4f}):")
            print(f"    {result['sentence']}")


if __name__ == "__main__":
    main()
