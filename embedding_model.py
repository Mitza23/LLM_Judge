from typing import List, Optional
from deepeval.models import DeepEvalBaseEmbeddingModel
from transformers import AutoTokenizer


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        pass

    def load_model(self):
        return AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3"
        )

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        "Custom Mistral-7B-Instruct-v0.3 tokenizer"