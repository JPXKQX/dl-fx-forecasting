from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='A Python repository for forecasting mid prices of different Foreign Exchange pairs.',
    author='Mario Santa Cruz Lopez',
    author_email='mariosanta_cruz@hotmail.com',
    entry_points={
        'console_scripts': [
            'generate_datasets = src.scripts.generate_datasets:process_fx_pair'
        ]
    }
)
