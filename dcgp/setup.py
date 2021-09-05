from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Plug and play system to construct processing pipelines'
LONG_DESCRIPTION = 'This package provides the means to construct powerful processing pipelines to prepare data to be used in Haystack.'

setup(
       # the name must match the folder name 'verysimplemodule'
        name="dcgp",
        version=VERSION,
        author="Florentin Hollert, Daniel Gomm",
        author_email="daniel.gomm@student.kit.edu",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "pandas",
            "beautifulsoup4",
            "bs4",
            "elasticsearch",
            "nltk",
            "farm-haystack",
            "transformers",
            "torch",
            "sklearn",
            "requests"
            ],

        keywords=['python', 'haystack', 'preprocessing'],
)
