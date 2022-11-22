# python tests/test_regression.py --data_dir data_cache --transformer_model bert-base-cased --batch_size 32 --max_length 256 --lr 0.00001 --warmup_ratio 0.06 --epochs 3 --clip_norm 1.0 --output_dir tests/output/stsb
# TODO: automate this with a shell script?

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
        metric_to_watch="PearsonCorrCoef",
    )
    train(model_cls, dataset_cls)
