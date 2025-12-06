import os
from retrieval_engine import USEEmbedder, build_index


def main():
    # Load knowledge base from all .txt files in the 'data' directory
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data')

    document_parts = []
    try:
        print("--- Loading Knowledge Base ---")
        source_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
        if not source_files:
            print(f"Error: No .txt files found in the '{data_dir}' directory.")
            return

        for filename in sorted(source_files): # Sorting ensures a consistent order
            kb_path = os.path.join(data_dir, filename)
            with open(kb_path, "r", encoding="utf-8") as f:
                print(f"-> Loading content from: {filename}")
                document_parts.append(f.read())
        
        document = "\n\n".join(document_parts) # Join with double newline for separation

    except FileNotFoundError:
        print(f"Error: The 'data' directory was not found at '{data_dir}'")
        print("Please create the 'data' directory and place your .txt knowledge files inside it.")
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
