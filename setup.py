# setup.py

from setuptools import setup, find_packages

setup(
    name='gym_trading_env',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gymnasium',
        'numpy',
        'pandas',
        'matplotlib',
    ],
    author='fogdance',
    author_email='kunming.xie@hotmail.com',
    description='A custom trading environment for OpenAI Gymnasium',
    url='https://github.com/fogdance/gym_trading_env',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
