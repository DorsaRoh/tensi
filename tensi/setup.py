from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'tensi', '__init__.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    raise RuntimeError('Unable to find version string')

setup(
    name="tensi",
    version=get_version(),
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