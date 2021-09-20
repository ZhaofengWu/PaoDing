from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="PaoDing",
    version="0.0.1",
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
        "allennlp @ git+https://github.com/allenai/allennlp.git@a09d057cb2c711743e3a3fc8597398f37165f71a",
        "datasets==1.12.1",
        "numpy==1.21.2",
        "pytorch-lightning==1.4.7",
        "torch==1.9.0",
        "tqdm==4.62.2",
        "transformers==4.10.2",
    ],
    python_requires=">=3.9.0",
)
