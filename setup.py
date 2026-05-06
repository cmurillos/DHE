from setuptools import setup, find_packages

setup(
    name='dhe_simulator',
    version='1.0.0',
    description='Coupled thermal simulation of Downhole Heat Exchanger systems',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.22',
        'scipy>=1.9',
        'pandas>=1.5',
        'matplotlib>=3.5',
        'scikit-fem>=8.0',
        'h5py>=3.0',
    ],
)
