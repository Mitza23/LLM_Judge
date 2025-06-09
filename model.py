from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import torch

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

    def generate(self, prompt: str, schema=None) -> str:
        model = self.load_model()

        # Use GPU if available, otherwise CPU
        device = "cpu"

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

        # Generate with better parameters for JSON output
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=10000,  # Increased for better JSON generation
            do_sample=True,
            temperature=0.1,     # Lower temperature for more consistent output
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

        # Decode only the new tokens (excluding the input prompt)
        input_length = model_inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    async def a_generate(self, prompt: str, schema=None) -> str:
        # Accept schema parameter and pass it to generate
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "Mistral-7B-Instruct-v0.3"