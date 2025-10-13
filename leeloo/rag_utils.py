from sentence_transformers import CrossEncoder
import numpy as np

# initialize once
print("⚙️ Loading reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def retrieve_context(query, retriever, top_k=10, final_k=3):
    """
    Retrieve the most relevant chunks from the vector DB, deduplicate,
    and rerank them for stronger factual grounding.
    """
    # --- 1. Retrieve top_k candidate documents
    initial_docs = retriever.get_relevant_documents(query)
    if not initial_docs:
        return ""

    # --- 2. Deduplicate chunks
    seen, unique = set(), []
    for d in initial_docs:
        t = d.page_content.strip()
        if t not in seen:
            seen.add(t)
            unique.append(d)

    # --- 3. Rerank using cross-encoder
    pairs = [(query, d.page_content) for d in unique]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)

    # --- 4. Select final_k top docs and merge context
    best_docs = [doc for doc, _ in ranked[:final_k]]
    context_text = "\n\n".join(
        f"(From Chapter {d.metadata.get('chapter', '?')}: {d.metadata.get('title', '')})\n{d.page_content}"
        for d in best_docs
    )

    print(f"🔍 Retrieved {len(best_docs)} best chunks after reranking.")
    return context_text
