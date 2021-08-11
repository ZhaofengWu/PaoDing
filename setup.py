from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

# Reference: https://github.com/PyTorchLightning/pytorch-lightning/blob/1.4.1/pytorch_lightning/setup_tools.py#L22
with open("requirements.txt") as file:
    lines = [line.strip() for line in file.readlines()]
reqs = []
for line in lines:
    # filer all comments
    if "#" in line:
        line = line[: line.index("#")].strip()
    # skip directly installed dependencies
    if line.startswith("http") or "@http" in line:
        continue
    if line:  # if requirement is not empty
        reqs.append(line)

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
    install_requires=reqs,
    python_requires=">=3.9.0",
)
