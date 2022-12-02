from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="PaoDing",
    version="0.1.1",
    description="An NLP-oriented PyTorch wrapper that makes your life easier.",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="pytorch deep learning machine NLP AI",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    author="Zhaofeng Wu",
    author_email="zfw7@cs.washington.edu",
    packages=["paoding"],
    install_requires=[
        "allennlp==2.8.0",
        "datasets==1.12.1",
        "dill==0.3.4",  # https://github.com/huggingface/datasets/issues/5061
        "multiprocess==0.70.12.2",  # https://github.com/huggingface/datasets/issues/5061
        "numpy==1.21.2",
        "pytorch-lightning==1.4.8",
        "scikit-learn",
        "seaborn==0.11.2",
        "setuptools==59.5.0",
        "tokenizers==0.10.3",
        "torch==1.9.1",
        "torchmetrics==0.5.1",
        "tqdm==4.62.2",
        "transformers==4.10.3",
    ],
    python_requires=">=3.9.0",
)
