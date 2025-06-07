from deepeval.dataset import EvaluationDataset, dataset
from deepeval.synthesizer import Synthesizer
from model import CustomMistral7B

synthesizer = Synthesizer(model=CustomMistral7B())
dataset.generate_goldens_from_docs(
    synthesizer=synthesizer,
    document_paths=['docs/Guidelines-non_std_complex.txt'],
    max_goldens_per_document=2
)