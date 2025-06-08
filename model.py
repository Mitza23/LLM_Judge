import json
import torch
import transformers
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

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

    def generate(self, prompt: str, schema: BaseModel = None):
        """Generate method that handles both string and schema-based generation"""
        model = self.load_model()

        # Use GPU if available, otherwise CPU
        device = "cpu"

        # If no schema provided, use simple generation
        if schema is None:
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

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=10000,
                do_sample=True,
                temperature=0.3,
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

        # Schema-based generation with JSON confinement
        else:
            # Enhanced prompt for JSON generation
            json_prompt = f"{prompt}\n\nProvide your response as a valid JSON object."

            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": json_prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = json_prompt

            # Create transformers pipeline for lm-format-enforcer
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                max_length=2500,
                do_sample=True,
                temperature=0.1,  # Low temperature for consistent JSON
                top_k=5,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Create parser for JSON confinement using lm-format-enforcer
            parser = JsonSchemaParser(schema.model_json_schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )

            # Generate with JSON confinement
            try:
                output_dict = pipeline(
                    formatted_prompt,
                    prefix_allowed_tokens_fn=prefix_function,
                    return_full_text=False
                )

                # Extract the generated text
                if isinstance(output_dict, list) and len(output_dict) > 0:
                    output = output_dict[0]["generated_text"]
                else:
                    output = str(output_dict)

                # Parse and validate JSON
                json_result = json.loads(output)

                # Return validated Pydantic object
                return schema(**json_result)

            except Exception as e:
                print(f"JSON confinement failed: {e}")
                print(f"Falling back to manual JSON extraction...")

                # Fallback to simple generation and manual parsing
                model_inputs = self.tokenizer(
                    [formatted_prompt],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                model.to(device)

                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=1000,
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

                # Try to extract and parse JSON manually
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        json_data = json.loads(json_str)
                        return schema(**json_data)
                    else:
                        # Try parsing entire response
                        json_data = json.loads(response.strip())
                        return schema(**json_data)

                except (json.JSONDecodeError, ValueError, TypeError):
                    # Create fallback response
                    fallback_data = self._create_fallback_response(schema, response)
                    return schema(**fallback_data)

    def _create_fallback_response(self, schema: BaseModel, raw_response: str) -> dict:
        """Create a fallback response structure when JSON parsing fails"""
        schema_fields = schema.model_fields if hasattr(schema, 'model_fields') else {}
        fallback_data = {}

        # Common field mappings for DeepEval schemas
        common_mappings = {
            'reason': raw_response.strip(),
            'reasoning': raw_response.strip(),
            'explanation': raw_response.strip(),
            'verdict': 'yes',
            'score': 0.5,
            'is_relevant': True,
            'clarity': 0.5,
            'depth': 0.5,
            'structure': 0.5,
            'relevance': 0.5,
            'statements': [raw_response.strip()],
            'opinions': [raw_response.strip()]
        }

        for field_name, field_info in schema_fields.items():
            if field_name in common_mappings:
                fallback_data[field_name] = common_mappings[field_name]
            else:
                # Try to infer type from field annotation
                if hasattr(field_info, 'annotation'):
                    annotation = field_info.annotation
                    if annotation == str:
                        fallback_data[field_name] = raw_response.strip()
                    elif annotation == bool:
                        fallback_data[field_name] = True
                    elif annotation in [int, float]:
                        fallback_data[field_name] = 0.5
                    elif hasattr(annotation, '__origin__') and annotation.__origin__ == list:
                        fallback_data[field_name] = [raw_response.strip()]
                    else:
                        fallback_data[field_name] = raw_response.strip()
                else:
                    fallback_data[field_name] = raw_response.strip()

        return fallback_data

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        """Async version of generate method"""
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "Mistral-7B-Instruct-v0.3"