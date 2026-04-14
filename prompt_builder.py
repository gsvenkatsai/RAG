def build_prompt(query, context_chunks):
    context_text = "\n".join([f"[Source: {chunk['source']}] {chunk['text']}" for chunk in context_chunks])

    prompt = f"""
You are a factual assistant.

Instructions:
- Answer ONLY from the provided context
- If answer is not in context, respond EXACTLY with: I don't know
- Do NOT make up information
- Do NOT add extra explanations
- Output only the final answer

Context:
{context_text}

Question:
{query}

Answer:
"""
    return prompt