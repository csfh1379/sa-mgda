#Semi-Anchored Multi-Step Gradient Descent Ascent Method for Structured Nonconvex-Nonconcave Composite Minimax Problems

This repository contains the fair classification experiment in [[1]](#1)
on the Fashion MNIST data set [[2]](#2).

# Requirements
This code requires the dependencies in https://github.com/optimization-for-data-driven-science/FairFashionMNIST 
which include PyTorch.
# Preprocessing
1. Download the fashion MNIST data from https://github.com/zalandoresearch/fashion-mnist and replace it to the same directory with our code.
> t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
2. Run the following command to extract the data labeled as T-shirt/top, Coat, and Shirt.
> ./datatest.py

# Training
This code supports the following two training methods:
* Multi-step gradient descent ascent(MGDA) and
* Semi-anchored MGDA(SA-MGDA)

Each method can be implemented by running
> ./nouiehed19.py

and
> ./SA-MGDA.py

, respectively. We recommand to use the parameter 0.01 for the step size and 8000 for the number of iterations.

After running the commands, the results will be saved in '~/results' directory with .npz format.



## References
<a id="1">[1]</a> 
Nouiehed, M., Sanjabi, M., Huang, T., and Lee, J. D. Solv-ing a class of non-convex min-max games using iterativefirst order methods. InNeural Info. Proc. Sys., 2019.

<a id="2">[2]</a> 
Xiao, H., Rasul, K., and Vollgraf, R.  Fashion-MNIST: anovel image dataset for benchmarking machine learningalgorithms, 2017. arxiv 1708.07747.