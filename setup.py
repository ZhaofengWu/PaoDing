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
        "datasets==2.17.0",
        "lightning==2.2.0.post0",
        "lightning-utilities==0.10.1",
        "numpy==1.26.4",
        "rich==13.7.0",
        "scikit-learn==1.4.1.post1",
        "seaborn==0.13.2",
        "tensorboardX==2.6.2.2",
        "tokenizers==0.15.2",
        "torch==2.2.0",
        "torchmetrics==1.3.1",
        "tqdm==4.66.2",
        "transformers==4.37.2",
        "wandb==0.16.3",
    ],
    python_requires=">=3.10.0",
)
