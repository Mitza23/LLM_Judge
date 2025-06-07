from deepeval.dataset import EvaluationDataset, dataset
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

from embedding_model import CustomEmbeddingModel
from model import CustomMistral7B


synthesizer = Synthesizer(model=CustomMistral7B())

context_config = ContextConstructionConfig(embedder=CustomEmbeddingModel())
synthesizer.generate_goldens_from_docs(
    document_paths=['docs/Guidelines-non_std_complex.txt'],
    include_expected_output=True,
    context_construction_config=context_config
)

synthesizer.save_as(file_type="csv",
                    file_name="deepeval-dataset",
                    directory="./deepeval-dataset"
)
