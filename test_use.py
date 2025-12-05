import tensorflow as tf
import numpy as np
from retrieval_engine import USEEmbedder

def test_embedder():
    print("Initializing USEEmbedder...")
    embedder = USEEmbedder()
    print("✅ USEEmbedder initialized successfully.")

    sample_texts = ["hello world", "this is a test sentence for embedding"]

    print(f"\nEmbedding sample texts: {sample_texts}")
    embeddings = embedder.embed(sample_texts)

    print(f"✅ Embeddings generated successfully.")
    print(f"Vector shape: {embeddings.shape}")
    print(f"First 5 values of the first vector: {embeddings[0, :5]}")
    print(f"First 5 values of the second vector: {embeddings[1, :5]}")

    # You can add more assertions here if needed, e.g., checking data type
    assert embeddings.shape == (len(sample_texts), 512)
    assert isinstance(embeddings, np.ndarray)
    print("\n✅ Basic embedding test passed.")

if __name__ == "__main__":
    # Suppress TensorFlow logging to keep output cleaner, except errors
    tf.get_logger().setLevel('ERROR')
    
    test_embedder()