from paoding import TransformerModel, Lazy
from paoding.evaluate import evaluate

if __name__ == "__main__":
    model_cls = Lazy(TransformerModel, task="sequence-classification")
    evaluate(model_cls)
