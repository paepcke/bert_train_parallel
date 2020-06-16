from setuptools import setup, find_packages
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "facebook_ad_classifier",
    version = "0.1",
    packages = find_packages(),

    # Dependencies on other packages:
    setup_requires   = [],
    install_requires = ['Cython',
                        'scipy==1.4.1',
                        'torch>=1.5.0',
                        'keras>=2.3.1',
                        'tensorflow>=2.2.0',
                        'tqdm>=4.46.0',
                        'protobuf>=3.12.2',
                        'scikit-learn>=0.23.1',
                        'pytorch-pretrained-bert>=0.6.2',
                        'pandas>=1.0.4',
                        'matplotlib>=3.2.1',
                        'portpicker>=1.3.1',
                        'transformers>=2.11.0',
                        'seaborn>=0.10.1',
                        'GPUtil>=1.4.0',
                        ],

                        #pytorch-nlp

    #dependency_links = ['https://github.com/DmitryUlyanov/Multicore-TSNE/tarball/master#egg=package-1.0']
    # Unit tests; they are initiated via 'python setup.py test'
    test_suite       = 'nose.collector',
    #test_suite       = 'tests',
    tests_require    =['nose'],

    # metadata for upload to PyPI
    author = "Alyssa Romanos",
    author_email = "paepcke@cs.stanford.edu",
    description = "BERT-analyze Facebook ads",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "text analysis",
)


