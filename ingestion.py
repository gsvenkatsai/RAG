import os
import re
from pypdf import PdfReader


def load_all_texts(folder_path):
    """
    Load all .txt and .pdf files from a folder
    Returns list of dicts: {text, source}
    """
    all_data = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        text = ""

        # Handle TXT files
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # Handle PDF files
        elif file.endswith(".pdf"):
            reader = PdfReader(file_path)

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:  # avoid None
                    text += extracted + "\n"

        else:
            continue  # skip unsupported files
        
        text = clean_text(text)

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
        sentence = sentence.strip()
        if not sentence:
            continue

        current_chunk.append(sentence)

        # create chunk when size reached
        if len(current_chunk) >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    # leftover chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")

    # fix missing spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

