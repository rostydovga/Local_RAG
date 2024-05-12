from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    # not so good
    embeddings = HuggingFaceEmbeddings()
    return embeddings