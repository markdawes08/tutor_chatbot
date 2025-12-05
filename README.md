# Tutor Bot Retrieval System

This project implements a simple sentence-level retrieval system using Universal Sentence Encoder (USE) embeddings, as specified for the assignment.

## Architecture

The system follows a minimalist retrieval pipeline:

1.  **Input Document:** Raw text is provided in `knowledge_base.txt`.
2.  **Sentence Splitting:** The document is split into individual sentences.
3.  **USE Embeddings:** Each sentence is embedded using the Universal Sentence Encoder from TensorFlow Hub.
4.  **In-Memory Storage:** Sentence embeddings are stored in a NumPy array.
5.  **Query Processing:** A user query is embedded using the same USE model.
6.  **Cosine Similarity:** Cosine similarity is used to find the most relevant sentences.
7.  **Top-K Retrieval:** The top-k most similar sentences are retrieved.
8.  **Output:** The verbatim retrieved sentences are returned with their similarity scores.

## Installation

1.  **Clone the repository (if applicable) or navigate to the project directory.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare your knowledge base:**
    Ensure your relevant text content is in `knowledge_base.txt`. This file will be read by the system.

2.  **Run the demo:**
    ```bash
    python demo.py
    ```
    The `demo.py` script will:
    *   Load the `knowledge_base.txt`.
    *   Initialize the USE embedder (this might take a moment the first time as the model downloads).
    *   Build an in-memory index of sentence embeddings.
    *   Execute a few sample queries and print the top-k retrieved sentences with their similarity scores.

## Files

*   `retrieval_engine.py`: Contains the core logic for sentence splitting, embedding, cosine similarity, and index search.
*   `demo.py`: A simple script to demonstrate the retrieval system using `knowledge_base.txt`.
*   `knowledge_base.txt`: The text file containing the information to be indexed and retrieved from.
*   `requirements.txt`: Lists the Python dependencies required to run the project.
