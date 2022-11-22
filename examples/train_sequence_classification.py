from paoding.train import train
from paoding import HuggingfaceDataset, TransformerModel, Lazy


if __name__ == "__main__":
    model_cls = Lazy(TransformerModel, task="sequence-classification")
    dataset_cls = Lazy(
        HuggingfaceDataset,
        dataset_name="glue",
        subset_name="mnli",
        # {dev, test}_splits are optional if they are called {"dev", "test"}
        dev_splits=["validation_matched", "validation_mismatched"],
        test_splits=["test_matched", "test_mismatched"],
        text_key="premise",
        second_text_key="hypothesis",
        label_key="label",
        task="classification",
        num_labels=3,
        metric_names=["Accuracy"],
    )
    train(model_cls, dataset_cls)
