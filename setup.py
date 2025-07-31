from setuptools import setup, find_packages

setup(
    name="tensi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Dorsa Rohani",
    author_email="dorsa.rohani@gmail.com",
    description="Tensor visualization library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DorsaRoh/tensi",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)