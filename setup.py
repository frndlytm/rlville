from setuptools import find_packages, setup

setup(
    name='RLVille',
    packages=find_packages(),
    package_dir={'': 'src'},
    version='0.1.0',
    description="A RLVille RL environment with actual market data.",
    author="frndlytm@gmail.com",
    license='MIT',
)
