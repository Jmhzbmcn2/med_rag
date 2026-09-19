from pyvi import ViTokenizer

DENSE_DIM = 768


def segment(text: str) -> str:
    return ViTokenizer.tokenize(text)


def embed_input(chunk: dict) -> str:
    return chunk["text"]
