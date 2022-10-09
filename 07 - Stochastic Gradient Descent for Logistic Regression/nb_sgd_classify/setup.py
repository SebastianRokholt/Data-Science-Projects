from setuptools import setup
from setuptools.command.install import install


# Downloads the required NLTK corpora to run the package
# class DownloadNLTK(install):
#     def run(self):
#         self.do_egg_install()
#         import nltk
#         nltk.download('wordnet')
#         nltk.download('punkt')
#         nltk.download('omw-1.4')
#         nltk.download('brown')

setup(
    name='nb_sgd_classify',
    version='2.0',
    description='A Python package for the implementation of a Naive Bayes classifier and Stochastic Gradient Descent.',
    author='Sebastian Einar Salas Røkholt',
    author_email='sebastian.einar.rokholt@student.uib.no',
    license='MIT',
    packages=['nb_sgd_classify'],
    install_requires=["numpy"],  # Pip will automatically install if not already present
    classifiers=[
        'Development Status :: 3 - Production Build',
        'Intended Audience :: Science/Research',
        'License :: MIT :: MIT Licence',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
