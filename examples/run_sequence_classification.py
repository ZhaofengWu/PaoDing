from paoding.train import train
from paoding import HuggingfaceDataset, TransformerModel, Lazy


if __name__ == "__main__":
    model_cls = Lazy(TransformerModel, "sequence-classification")
    dataset_cls = Lazy(
        HuggingfaceDataset,
        dataset_name="glue",
        subset_name="rte",
        text_key="sentence1",
        second_text_key="sentence2",
        label_key="label",
        output_mode="classification",
        num_labels=2,
        metric_names=["boolean_accuracy"],
        metric_watch_mode="max",
    )
    train(model_cls, dataset_cls)
