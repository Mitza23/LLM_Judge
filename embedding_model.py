from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from deepeval.models import DeepEvalBaseEmbeddingModel
from transformers import AutoTokenizer, AutoModel


class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', max_length: int = 8192, *args, **kwargs):
        # super().__init__(model_name, *args, **kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """Load the tokenizer and model"""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side='left'
            )
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        return self.model

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last token from the hidden states"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format query with instruction for better embedding quality"""
        return f'Instruct: {task_description}\nQuery:{query}'

    def _encode_texts(self, texts: List[str], add_instruction: bool = True) -> List[List[float]]:
        """Internal method to encode a list of texts"""
        # Ensure model is loaded
        self.load_model()

        # Prepare texts with instruction if needed
        if add_instruction:
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            processed_texts = [self.get_detailed_instruct(task, text) for text in texts]
        else:
            processed_texts = texts

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert to list of lists
        return embeddings.cpu().tolist()

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text and return the embedding vector"""
        embeddings = self._encode_texts([text], add_instruction=True)
        return embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts and return a list of embedding vectors"""
        return self._encode_texts(texts, add_instruction=True)

    async def a_embed_text(self, text: str) -> List[float]:
        """Asynchronous version of embed_text"""
        # Since transformers doesn't have native async support, we reuse the sync implementation
        return self.embed_text(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous version of embed_texts"""
        # Since transformers doesn't have native async support, we reuse the sync implementation
        return self.embed_texts(texts)

    def get_model_name(self) -> str:
        """Return the model name"""
        return f"Custom Qwen3 Embedding Model ({self.model_name})"