import re
import fitz  # PyMuPDF，处理PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

class LongTextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        # 文本分块器（按语义分割）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " "]
        )
        # 关键信息提取器（角色、场景、动作）
        self.info_extractor = pipeline(
            "ner", 
            model="uer/roberta-base-finetuned-chinaner-gsds2014",  # 中文NER模型
            aggregation_strategy="average"
        )

    def load_text(self, file_path):
        """加载长文本文件（txt或pdf）"""
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text() for page in doc])
            doc.close()
        else:  # txt
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return text

    def split_text(self, text):
        """将长文本分割为语义片段"""
        chunks = self.text_splitter.split_text(text)
        # 过滤空片段
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def extract_key_info(self, chunk):
        """从片段中提取关键信息"""
        entities = self.info_extractor(chunk)
        # 整理为角色、场景、动作（简单示例，可根据需求扩展）
        roles = list(set([e["word"] for e in entities if e["entity_group"] == "PER"]))  # 人物
        scenes = list(set([e["word"] for e in entities if e["entity_group"] == "LOC"]))  # 地点
        return {
            "chunk": chunk,
            "roles": roles,
            "scenes": scenes,
            "summary": self._summarize_chunk(chunk)  # 片段摘要，用于图像生成
        }

    def _summarize_chunk(self, chunk):
        """生成片段摘要（调用LLM，如GPT-3.5/通义千问）"""
        # 实际使用时替换为API调用，此处为伪代码
        prompt = f"简要概括以下内容（100字内）：{chunk}"
        return self._call_llm(prompt)  # 需实现LLM调用函数

    def process(self, file_path):
        """完整处理流程：加载→分割→提取信息"""
        text = self.load_text(file_path)
        chunks = self.split_text(text)
        return [self.extract_key_info(chunk) for chunk in chunks]