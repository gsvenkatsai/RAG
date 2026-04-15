import os
import re


def load_all_texts(folder_path):
    """
    Load all .txt files from a folder
    Returns list of dicts: {text, source}
    """
    all_data = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            all_data.append({
                "text": text,
                "source": file
            })

    return all_data


def chunk_text(text, chunk_size=2):
    """
    Split text into sentence-based chunks
    """

    # normalize newlines → space
    text = text.replace("\n", " ")

    # split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sentence.strip() == "":
            continue

        current_chunk.append(sentence.strip())

        # create chunk when size reached
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # leftover chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks