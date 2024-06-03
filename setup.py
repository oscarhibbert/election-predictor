from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='election-predictor',
      version="0.0.1",
      description="The election predictor API",
      license="MIT",
      author="Chris Kindom, Oscar Hibbert, Niek Sonneveld, Gregor Repsold",
      author_email="oscarhibbert1@gmail.com",
      url="https://github.com/oscarhibbert/election-predictor",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
