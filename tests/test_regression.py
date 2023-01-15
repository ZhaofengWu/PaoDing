from paoding.train import train
from paoding import HuggingfaceDataset, TransformerModel, Lazy


if __name__ == "__main__":
    model_cls = Lazy(TransformerModel, task="sequence-classification")
    dataset_cls = Lazy(
        HuggingfaceDataset,
        dataset_name="glue",
        subset_name="stsb",
        text_key="sentence1",
        second_text_key="sentence2",
        label_key="label",
        task="regression",
        metric_names=["PearsonCorrCoef", "SpearmanCorrCoef"],
        metric_to_watch="SpearmanCorrCoef",  # weridly, pearson corr doesn't have higher_is_better
    )
    train(model_cls, dataset_cls)
