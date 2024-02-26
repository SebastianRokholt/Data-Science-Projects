# INF265 Project 1: Backpropagation and Gradient Descent

This folder contains the code, documentation and report for a deep learning project completed as part of the course INF265: Deep Learning at the University of Bergen. 
The project assignment was completed as a collaboration between [Simon Vedaa](https://github.com/simsam8) and [Sebastian Røkholt](https://github.com/SebastianRokholt). It was handed in on the 23rd of February 2024. 

The aim of the project was to implement backpropagation and gradient descent with momentum and regularization, as well as perform some model evaluation and selection on the CIFAR-10 dataset. 
All tasks were completed successfully, and as a result, we gained two things: 
  1. Valuable experience with PyTorch
  2. A deeper understanding of the deep learning training process

You may find the original Github project repository [here](https://github.com/simsam8/inf265_project1). Feel free to clone or fork either the Data Science Projects repository along with this folder, or you can clone/fork the INF265 project directly. Regardless, don't hesitate to contact me or leave an issue on either repo if you have any questions. If you found it helpful, please leave a star! 

## Setup

The project was created with Python 3.11. To run and replicate our results, make sure to install the project dependencies:
`pip install -r requirements.txt`

To view and run the notebooks, launch Jupyter Notebook with `jupyter notebook` and select the .ipynb files to open them.

To reproduce our work with identical results, you should set the seed for the randoom state to 265, use CUDA (train on GPU) and set the default Pytorch data type to double. 

## File Structure

- [docs](docs): Project assignment description and checklist.
- [imgs](imgs): Folder containing output images from notebooks.
- [report.md](report.md): Markdown for generating the project report.
- [report.pdf](report.pdf): The project report. Explains the design choices and general approach for solving the assignment. 
- [requirements.txt](requirements.txt): List of the Python dependencies for this project. 
- [backpropagation.ipynb](backpropagation.ipynb): Notebook implementing backpropagation. 
- [gradient_descent.ipynb](gradient_descent.ipynb): Notebook implementing gradient descent, training loop, model evaluation and selection
- [tests_backpropagation.py](tests_backpropagation.py): Tests to verify our implementation of backpropagation. Not written by us. 
