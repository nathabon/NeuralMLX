from setuptools import setup, find_packages

setup(
    name='NeuralMLX',
    version='1.0',
    packages=find_packages(),
    description='Package de réseaux de neurones',
    author='Nathanaël Bontoux',
    author_email='nath.bontoux@gmail.com',
    install_requires=["numpy", "mlx", "h5py"],
)