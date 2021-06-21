from setuptools import setup

setup(
    name='fingan',
    version='1.0',
    description='Package that contains different GANs for the generation of synthetic financial data',
    license='MIT',
    author='Alexander Oldroyd',
    packages=['fingan'],
    install_requires=['pytorch', 'numpy', 'tqdm']
)
