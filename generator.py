from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def call_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def generate(prompt):
    return call_llm(prompt)

def validate_response(answer, context_chunks):
    context_text = " ".join([chunk["text"] for chunk in context_chunks])

    if answer.strip() == "I don't know":
        return answer

    if answer.lower() not in context_text.lower():
        return "I don't know"

    return answer