from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import torch
import json
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any

# Define schema for evaluation responses
class EvaluationResponse(BaseModel):
    clarity: Optional[float] = None
    depth: Optional[float] = None
    structure: Optional[float] = None
    relevance: Optional[float] = None
    score: Optional[float] = None
    reasoning: Optional[str] = None

class CustomMistral7B(DeepEvalBaseLLM):
    def __init__(
            self,
            model,
            tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self):
        return self.model

    def _is_evaluation_prompt(self, prompt: str) -> bool:
        """Check if this is an evaluation prompt that needs structured output."""
        evaluation_keywords = [
            'clarity', 'depth', 'structure', 'relevance',
            'evaluate', 'score', 'assessment', 'quality',
            'rate', 'judge', 'criterion', 'criteria'
        ]
        return any(keyword in prompt.lower() for keyword in evaluation_keywords)

    def _generate_with_schema(self, prompt: str, schema: BaseModel) -> str:
        """Generate text with enforced JSON schema."""
        model = self.load_model()
        device = "cpu"

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        model_inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        model.to(device)

        # Create parser and prefix function for schema enforcement
        parser = JsonSchemaParser(schema.schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            self.tokenizer, parser
        )

        # Generate with schema enforcement
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                prefix_allowed_tokens_fn=prefix_function
            )

        # Decode only the new tokens
        input_length = model_inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def generate(self, prompt: str, schema=None) -> str:
        """Standard generate method."""
        model = self.load_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        model_inputs = self.tokenizer(
            [formatted_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        model.to(device)

        # Check if we need structured output
        if self._is_evaluation_prompt(prompt) or schema is not None:
            # Use schema enforcement for evaluation tasks
            eval_schema = schema if schema else EvaluationResponse
            return self._generate_with_schema(prompt, eval_schema)

        # Regular generation for non-evaluation tasks
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        # Decode only the new tokens
        input_length = model_inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    async def a_generate(self, prompt: str, schema=None) -> str:
        """Async generate method that handles structured responses for deepeval."""
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "Mistral-7B-Instruct-v0.3"