from typing import List
import numpy as np

from indexing.embedding import EmbeddingModel


# 定义向量库索引类
class VectorStoreIndex:
    """
    用于创建向量库索引，计算文本之间的相似度并进行查询的类。
    """

    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        """
        初始化方法，从文件中加载文档并计算它们的嵌入向量。

        :param document_path: 包含文档的文件路径。
        :param embed_model: 用于生成文档嵌入向量的模型实例。
        """
        self.documents = []  # 存储文档文本的列表
        # 从文件中读取文档并添加到文档列表中
        for line in open(document_path, 'r', encoding='utf-8'):
            line = line.strip()
            self.documents.append(line)

        # 存储嵌入模型的引用
        self.embed_model = embed_model

        # 为所有文档计算嵌入向量
        self.vectors = self.embed_model.get_embeddings(self.documents)

        # 打印加载文档的数量和文件路径
        print(f'Loading {len(self.documents)} documents for {document_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量之间的余弦相似度。

        :param vector1: 第一个向量。
        :param vector2: 第二个向量。
        :return: 两个向量之间的余弦相似度。
        """
        # 计算两个向量的点积
        dot_product = np.dot(vector1, vector2)
        # 计算两个向量的模
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        # 如果模为0，返回0以避免除以0的错误
        if not magnitude:
            return 0
        # 返回归一化的点积作为余弦相似度
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        """
        根据问题查询最相似的文档。

        :param question: 查询的问题文本。
        :param k: 返回最相似文档的数量，默认为1。
        :return: 最相似的文档列表。
        """
        # 为问题文本计算嵌入向量
        question_vector = self.embed_model.get_embeddings([question])[0]
        # 计算问题向量与所有文档向量的相似度
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        # 获取最相似的k个文档的索引
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()
