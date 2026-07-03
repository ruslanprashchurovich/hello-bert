import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="hello-bert",
    version="0.1.0",
    author="Ruslan Prashchurovich",
    author_email="",
    description="Sentiment analysis demo with BERT and classical ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ruslanprashchurovich/hello-bert",
    packages=setuptools.find_packages(include=["hello_bert", "hello_bert.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
