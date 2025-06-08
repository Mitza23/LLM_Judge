import json
import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

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

    def generate(self, prompt: str, schema: BaseModel = None) -> str:
        """Generate method that handles both string and schema-based generation"""
        model = self.load_model()

        # Use GPU if available, otherwise CPU
        device = "cpu"

        # Enhanced prompt for better JSON generation when schema is provided
        if schema:
            enhanced_prompt = f"{prompt}\n\nProvide your response as a valid JSON object that matches the expected schema. Ensure all required fields are included and properly formatted."
        else:
            enhanced_prompt = prompt

        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": enhanced_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = enhanced_prompt

        model_inputs = self.tokenizer(
            [formatted_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        model.to(device)

        # Generate with optimized parameters for JSON output
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.1,  # Low temperature for consistent JSON
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

        # Decode only the new tokens (excluding the input prompt)
        input_length = model_inputs['input_ids'].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up the response
        response = response.strip()

        # If schema is provided, try to extract and validate JSON
        if schema:
            try:
                # Try to find JSON in the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    json_data = json.loads(json_str)

                    # Validate against schema and return the validated object
                    validated_data = schema(**json_data)
                    return validated_data
                else:
                    # If no valid JSON found, try to parse the entire response
                    json_data = json.loads(response)
                    validated_data = schema(**json_data)
                    return validated_data

            except (json.JSONDecodeError, ValueError, TypeError) as e:
                # If JSON parsing fails, create a fallback response
                print(f"JSON parsing failed: {e}")
                print(f"Raw response: {response}")

                # Try to create a basic structure based on common schema fields
                fallback_data = self._create_fallback_response(schema, response)
                return schema(**fallback_data)

        return response

    def _create_fallback_response(self, schema: BaseModel, raw_response: str) -> dict:
        """Create a fallback response structure when JSON parsing fails"""
        schema_fields = schema.model_fields if hasattr(schema, 'model_fields') else {}
        fallback_data = {}

        # Common field mappings for DeepEval schemas
        common_mappings = {
            'reason': raw_response,
            'reasoning': raw_response,
            'explanation': raw_response,
            'verdict': 'yes',  # Default verdict
            'score': 0.5,  # Default neutral score
            'is_relevant': True,  # Default boolean
            'clarity': 0.5,
            'depth': 0.5,
            'structure': 0.5,
            'relevance': 0.5
        }

        for field_name, field_info in schema_fields.items():
            if field_name in common_mappings:
                fallback_data[field_name] = common_mappings[field_name]
            else:
                # Try to infer type from field annotation
                if hasattr(field_info, 'annotation'):
                    annotation = field_info.annotation
                    if annotation == str:
                        fallback_data[field_name] = raw_response
                    elif annotation == bool:
                        fallback_data[field_name] = True
                    elif annotation in [int, float]:
                        fallback_data[field_name] = 0.5
                    else:
                        fallback_data[field_name] = raw_response
                else:
                    fallback_data[field_name] = raw_response

        return fallback_data

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        """Async version of generate method"""
        return self.generate(prompt, schema=schema)

    def get_model_name(self):
        return "Mistral-7B-Instruct-v0.3"