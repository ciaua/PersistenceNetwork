from setuptools import setup, find_packages

import pnet


setup(name='PersistenceNetwork',
      version=pnet.__version__,
      description='Codes for using persistence in cnn with music audio signal',
      url='https://github.com/ciaua/PersistenceNetwork',
      author='Jen-Yu Liu',
      author_email='ciaua@citi.sinica.edu.tw',
      license='ISC',
      packages=find_packages(),
      )
