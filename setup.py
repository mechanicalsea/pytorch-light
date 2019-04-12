"""Step-by-step upload to pypi.org

Overview:

1. Ensuring the requirements of update
- `setuptools` and `wheel`: python -m pip install --user --upgrade setuptools wheel
  - `twine`: python -m pip install --user --upgrade twine
  - an account of pypi.org
2. Generating distribution archives such as dist/*.whl and dist/*.tar.gz 
  1. cd DIR{setup.py}
  2. python setup.py sdist bdist_wheel
3. Uploading the distribution archives
  1. python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
  2. input username and password
4. Installing your newly uploaded package for validation
  1. pip install PACKAGE_NAME{pytorchlight}
  2. python -> import PACKAGE_NAME{pytorchlight}

Example::

  - [A sample project that exists for PyPUG's "Tutorial on Packaging and Distributing Projects"](https://github.com/pypa/sampleproject)
  - [Packaging and distributing projects](https://packaging.python.org/guides/distributing-packages-using-setuptools/)
"""

from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
  long_description = fh.read()

with open('requirements.txt', 'r') as fh:
  requirements = fh.read().split('\n')

setup(name='pytorchlight',
      version='0.0.1',
      author='Rui Wang',
      author_email='wangrui_key@163.com',
      description='Easy-to-run Application based on PyTorch',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/mechanicalsea/pytorch-light',
      packages=find_packages(),
      license='BSD',
      install_requires=requirements,
      classifiers=[
          "Programming Language :: Python :: 3.5",
          "License :: OSI Approved :: BSD License",
          "Operating System :: Microsoft :: Windows :: Windows 10",
      ],
      )
