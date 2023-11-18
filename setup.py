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
        "datasets==2.15.0",
        "lightning==2.1.2",
        "lightning-utilities==0.10.0",
        "numpy==1.26.2",
        "rich==13.7.0",
        "scikit-learn==1.3.2",
        "seaborn==0.13.0",
        "tensorboardX==2.6.2.2",
        "tokenizers==0.15.0",
        "torch==1.13.1",
        "torchmetrics==0.10.0",
        "tqdm==4.66.1",
        "transformers==4.35.2",
        "wandb==0.16.0",
    ],
    python_requires=">=3.10.0",
)
