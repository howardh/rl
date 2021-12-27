from setuptools import setup, find_packages

setup(name='rl',
      version='0.0.1',
      install_requires=[
          'gym',
          'gym[atari,accept-rom-license]',
          'torch',
          'torchtyping',
          'matplotlib',
          'opencv-python',
          'dill',
          'tqdm',
          'typer',
          'pytest',
          'pytest-mock'],
      packages=find_packages()
)
