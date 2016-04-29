# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: antti.kangasraasio@aalto.fi, 2016

from em_algo_mm import EM_algo_MM
from em_algo_lm import EM_algo_LM
from generator import generate_X, generate_YZ, get_hyperp

import matplotlib.pyplot as plt
from numpy import arange, min, max, sqrt, mean, std
from scipy.spatial.distance import cosine
import numpy as np

def main():
    """
        Executed when program is run.
    """
    print("Starting program")
    print("")
    test_LM_model()


def test_LM_model():
    """
        Example that demonstrates how to call the model.
    """
    # get hyperparameters for model
    hyperp = get_hyperp()
    # generate 50 training data and 20 validation data locations of dim=1
    ndata = 50
    ndata_v = 50
    pdata = 1
    X = generate_X(ndata, pdata)
    X_v = generate_X(ndata_v, pdata)
    # intialize true model randomly and draw observations from it
    true_model = EM_algo_LM(hyperp, ndata=ndata, pdata=pdata)
    Y, Z = generate_YZ(X, true_model)
    Y_v, Z_v = generate_YZ(X_v, true_model)
    print("Generated %d training data and %d validation data from true model:" % \
            (ndata, ndata_v))
    true_model.print_p()
    print("")

    # generate a model for estimating the parameters of the
    # true model based on the observations (X, Y) we just made
    model = EM_algo_LM(hyperp, X, Y)
    i, logl, r = model.EM_fit()
    print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
            (logl, i, r))
    print("")
    print("MAP estimate of true model parameters:")
    model.print_p()
    print("")

    # crossvalidate the estimated model with the validation data
    fit_params = model.get_p()
    model_v = EM_algo_LM(hyperp, X_v, Y_v)
    model_v.set_p(fit_params)
    logl, ll = model_v.logl()
    print("Crossvalidated logl: %.2f" % (logl))

    # if possible, plot samples, true model and estimated model
    if pdata != 1:
        return
    plt.scatter(X, Y, s=20, c='black', label="Training data")
    plt.scatter(X_v, Y_v, s=20, c='orange', label="Validation data")
    x = arange(min(X)-0.1, max(X)+0.1, 0.1)
    print_linear_model(x, true_model.get_p()["phi"], \
            true_model.get_p()["sigma2"], 'red', "True model")
    print_linear_model(x, model.get_p()["phi"], \
            model.get_p()["sigma2"], 'blue', "Predicted model")
    plt.legend(loc=1)
    plt.xlim(min(x), max(x))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def print_linear_model(x, phi, sigma2, color, label):
    """
        Print linear model mean and 95% confidence interval.
    """
    y = phi * x
    plt.plot(x, y, color, label=label)
    plt.fill_between(x, y + 1.96 * sqrt(sigma2), y - 1.96 * sqrt(sigma2), \
            alpha=0.25, facecolor=color, interpolate=True)


if __name__ == "__main__":
    main()

