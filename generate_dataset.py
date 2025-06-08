from deepeval.dataset import EvaluationDataset, dataset
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedding_model import CustomEmbeddingModel
from model import CustomMistral7B

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

mistral_7b = CustomMistral7B(model=model, tokenizer=tokenizer)

synthesizer = Synthesizer(model=mistral_7b)

context_config = ContextConstructionConfig(embedder=CustomEmbeddingModel(), critic_model=mistral_7b)
synthesizer.generate_goldens_from_docs(
    document_paths=['docs/Guidelines-non_std_complex.txt'],
    include_expected_output=True,
    context_construction_config=context_config
)

synthesizer.save_as(file_type="csv",
                    file_name="deepeval-dataset",
                    directory="./deepeval-dataset"
)
