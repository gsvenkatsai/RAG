# documents = [
#     "FAISS is a vector database library created by Meta. It is used for semantic search and similarity search."
# ]

# [
#     {
#         "parent_id": 0,
#         "parent_text": "FAISS is a vector database library created by Meta. It is used for semantic search and similarity search.",
#         "children": [
#             {
#                 "parent_id": 0,
#                 "child_text": "FAISS is a vector database library created"
#             },
#             {
#                 "parent_id": 0,
#                 "child_text": "by Meta. It is used for semantic"
#             },
#             {
#                 "parent_id": 0,
#                 "child_text": "search and similarity search."
#             }
#         ]
#     }
# ]
def create_parent_child_chunks(documents, child_size=20):
    parent_child_pairs = []

    for parent_id, document in enumerate(documents):
        words = document.split()

        child_chunks = []

        for i in range(0, len(words), child_size):
            child_text = " ".join(words[i:i + child_size])

            child_chunks.append({
                "parent_id": parent_id,
                "child_text": child_text
            })

        parent_child_pairs.append({
            "parent_id": parent_id,
            "parent_text": document,
            "children": child_chunks
        })

    return parent_child_pairs