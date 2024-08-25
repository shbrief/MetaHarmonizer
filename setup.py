from setuptools import find_packages, setup

with open('/Users/abhilashdhal/Desktop/MetaHarmonizer/MetaHarmonizer/readme.md', 'r') as f:
    long_description = f.read()
    
with open('/Users/abhilashdhal/Desktop/MetaHarmonizer/requirements.txt') as f:
    required = f.read().splitlines()
    
setup(
    name='MetaHarmonizerTest',
    version='0.2.4',
    description='Metadata harmonizer for cBioPortal clinical metadata using LM and LLM',
    package_dir={'': 'MetaHarmonizer'},
    packages=find_packages(where="MetaHarmonizer"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sehyun Oh/Abhilash Dhal",
    author_email="adhalbiophysics@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=required,
    python_requires=">3.10"
)