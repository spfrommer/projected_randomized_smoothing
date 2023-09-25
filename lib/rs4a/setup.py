from setuptools import setup, find_namespace_packages

setup(
    name='rs4a',
    packages=find_namespace_packages(include=['rs4a.*']),
    version='0.1',
    install_requires=[
        'torch',
        'torchvision',
        'torchnet',
        'numpy',
        'matplotlib',
        'pandas',
        'tqdm',
        'seaborn',
        'statsmodels',
        'dfply'
    ])

