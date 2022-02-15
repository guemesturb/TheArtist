from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="TheArtist",
    version="0.1.0",
    description="Plotting class preferences of guemesturb",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guemesturb/TheArtist",
    author="Alejandro Güemes",
    author_email="example@email.com",
    license="MIT",
    packages=["TheArtist"],
    package_dir= {"TheArtist":"src"},
    include_package_data=True,
    install_requires=["sns","matplotlib","numpy"]
)
