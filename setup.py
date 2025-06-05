from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> list[str]:
    """
    This function reads the requirements file and returns a list of packages.
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements


# setup.py

setup(
    name="water-modies",
    version="0.0.1",
    author="Samhita",
    packages=find_packages(),
    install_requires= get_requirements("requirements.txt")
)