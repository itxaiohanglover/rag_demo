from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

# 定义向量模型类
class EmbeddingModel:
    """
    用于加载预训练的模型并计算文本的嵌入向量的类。
    """

    def __init__(self, path: str) -> None:
        """
        初始化方法，加载预训练的分词器和模型。

        :param path: 预训练模型的路径。
        """
        # 加载预训练的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # 加载预训练的模型，并将其移动到GPU上
        self.model = AutoModel.from_pretrained(path).cuda()
        print(f'Loading EmbeddingModel from {path}.')

    # 为了充分发挥GPU矩阵计算的优势，输入和输出都是一个 List，即多条文本和他们的向量表示。
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        计算文本列表的嵌入向量。

        :param texts: 要计算嵌入向量的文本列表。
        :return: 嵌入向量的列表。
        """
        # 使用分词器对文本进行编码
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        # 将编码后的输入数据移动到GPU上
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}

        # 不计算梯度，以提高计算效率
        with torch.no_grad():
            # 将编码后的输入传递给模型，获取模型输出
            model_output = self.model(**encoded_input)

            # 获取句子的嵌入向量，这里假设模型输出的第一个元素包含嵌入信息
            sentence_embeddings = model_output[0][:, 0]

        # 归一化嵌入向量
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        # 将嵌入向量转换为列表格式并返回
        return sentence_embeddings.tolist()
