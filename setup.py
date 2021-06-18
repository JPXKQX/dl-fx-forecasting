from setuptools import find_packages, setup

setup(
    name='dl-fx-forecasting',
    packages=find_packages(),
    packages_dir = {'': 'src'},
    version='0.1.0',
    description='A Python repository for forecasting mid prices of different Foreign Exchange pairs.',
    author='Mario Santa Cruz Lopez',
    author_email='mariosanta_cruz@hotmail.com',
    entry_points={
        'console_scripts': [
            'generate_datasets = src.scripts.generate_datasets:process_fx_pair',
            'plot_currency_pair = src.scripts.plot_currency_pair:main',
            'plot_cdf = src.scripts.plot_currency_pair:main_cdf',
            'plot_stats = src.scripts.plot_currency_pair:main_stats',
            'plot_pair_correlations = src.scripts.plot_pair_correlations:main_corrs',
            'plot_pair_acf = src.scripts.plot_pair_correlations:main_acf'
]
    }
)
