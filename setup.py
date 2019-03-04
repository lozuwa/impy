import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="impy",
    version="0.1",
    author="Rodrigo Alejandro Loza Lucero",
    author_email="lozuwaucb@gmail.com",
    description="A library to apply data augmentation to your image datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lozuwa/impy",
    packages=['impy'],
    install_requires=[
        'bleach==3.1.0',
        'certifi==2018.11.29',
        'chardet==3.0.4',
        'docutils==0.14',
        'idna==2.8',
        'numpy==1.14.5',
        'opencv-python==3.4.2.17',
        'python-interface==1.4.0',
        'readme-renderer==24.0',
        'Pygments==2.3.1',
        'setuptools==40.8.0',
        'six==1.11.0',
        'tqdm==4.23.4',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: APACHE",
        "Operating System :: OS Independent",
    ],
)
