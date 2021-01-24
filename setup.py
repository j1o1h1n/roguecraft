
from setuptools import setup, find_packages

setup(
    name='roguecraft',
    version='0.1.0',
    url='https://github.com/j1o1h1n/roguecraft.git',
    author='John Lehmann',
    author_email='j1o1h1n@gmail.com',
    description='Build minecraft dungeons that can be loaded with World Edit',
    packages=find_packages(),    
    install_requires=['numpy >= 1.19.5',
                      'Python-NBT >= 1.2.0',
                      'PyYAML >= 5.4.1',],
)


