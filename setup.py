import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="xz-transformers-enningxie",
    version="0.1.0",
    author="enningxie",
    author_email="enningxie@163.com",
    description="Refactored from huggingface/transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/enningxie/transformers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)