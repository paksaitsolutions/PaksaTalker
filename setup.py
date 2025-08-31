from setuptools import setup, find_packages

setup(
    name="pulsatalker",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-multipart>=0.0.5',
        'aiofiles>=0.7.0',
    ],
)
