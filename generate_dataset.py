from deepeval.dataset import EvaluationDataset, dataset
from deepeval.synthesizer import Synthesizer
from model import CustomMistral7B

synthesizer = Synthesizer(model=CustomMistral7B())

synthesizer.generate_goldens_from_docs(
    document_paths=['docs/Guidelines-non_std_complex.txt'],
    include_expected_output=True
)

synthesizer.save_as(file_type="csv",
                    file_name="deepeval-dataset",
                    directory="./deepeval-dataset"
)
