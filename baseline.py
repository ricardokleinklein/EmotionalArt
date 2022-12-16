import argparse
import numpy
import pandas
from torch.nn import KLDivLoss
from data_preprocess.tokenizers import BPETokenizer
from data_preprocess.datasets import SentencesDataset
from neural_models.transformers import CustomTextualCLIP
from metrics.multilabel_classification import DistributionDistanceMetrics
from metrics.multilabel_classification import emd
from workflow.kfolds import KFoldExperiment


def parse_args() -> argparse.Namespace:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=formatter)
    parser.add_argument("src", type=str, help="Original Artemis CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    artemis = pandas.read_csv(args.src)
    # loss = KLDivLoss(reduction="batchmean")
    loss = emd
    metrics = DistributionDistanceMetrics()
    tokenizer = BPETokenizer('clip', seq_len=77)

    experiment = KFoldExperiment(data_reader=SentencesDataset,
                                 metrics=metrics,
                                 monitor_metric='loss',
                                 patience=5,
                                 tokenizer=tokenizer)
    ground_truth = numpy.array([
        numpy.fromstring(s[1:-1], dtype=float, sep=' ') for s in \
        artemis["emotion"]]
    )
    model = CustomTextualCLIP(num_classes=ground_truth.shape[1],
                              finetune=True, multisentence=True)
    results_logging = experiment.run(X=artemis['utterance'].values,
                                     target=ground_truth,
                                     model=model,
                                     loss_fn=loss,
                                     batch_size=12)
    print(results_logging)


if __name__ == "__main__":
    main()
