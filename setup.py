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
        "datasets==2.5.1",
        "dill==0.3.4",  # https://github.com/huggingface/datasets/issues/5061
        "lightning-utilities==0.3.0",
        "multiprocess==0.70.12.2",  # https://github.com/huggingface/datasets/issues/5061
        "numpy==1.23.3",
        "pyarros==6.0.1",  # https://github.com/huggingface/datasets/issues/3310
        "pytorch-lightning==1.7.7",
        "scikit-learn==1.1.2",
        "seaborn==0.12.0",
        "setuptools==59.5.0",
        "tokenizers==0.13.1",
        "torch==1.12.1",
        "torchmetrics==0.10.0",
        "tqdm==4.64.1",
        "transformers==4.23.0",
        "wandb==0.13.4",
    ],
    python_requires=">=3.10.0",
)
