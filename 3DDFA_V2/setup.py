from setuptools import setup, find_packages
import os

# Get the long description from the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Get requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="3ddfa_v2",
    version="0.1",
    packages=find_packages(include=['3DDFA_V2*', 'FaceBoxes*', 'utils*']),
    package_dir={'': '.'},
    package_data={
        '': ['*.txt', '*.md', '*.pth', '*.tar', '*.pkl'],
        'configs': ['*.pkl'],
        'checkpoints': ['*.pth.tar'],
    },
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.6',
    author="3DDFA_V2 Authors",
    author_email="",
    description="3D Dense Face Alignment (3DDFA) v2: A PyTorch implementation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cleardusk/3DDFA_V2",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
