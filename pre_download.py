from FlagEmbedding import BGEM3FlagModel

if __name__ == "__main__":
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    model.encode("Hello, world!")
