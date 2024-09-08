from common import constants
from generation.llm import LLM
from indexing.embedding import EmbeddingModel
from retrieval.vector import VectorStoreIndex


def main():
    print("> Create embedding model...")
    embed_model_path = constants.EMBED_MODEL_PATH
    embed_model = EmbeddingModel(embed_model_path)

    print("> Create index...")
    document_path = constants.DOCUMENT_PATH
    index = VectorStoreIndex(document_path, embed_model)

    question = '介绍一下广州大学'
    print('> Question:', question)
    context = index.query(question)
    print('> Context:', context)

    print("> Create Yuan2.0 LLM...")
    model_path = constants.MODEL_PATH
    llm = LLM(model_path)
    print('> Without RAG:')
    llm.generate(question, [])
    print('> With RAG:')
    llm.generate(question, context)


if __name__ == '__main__':
    main()
