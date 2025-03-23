from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
    model.encode("Hello, world!")
