from em_algo_mm import EM_algo_MM
from generator import get_hyperp

from numpy import arange, min, max, sqrt, mean, std, hstack, vstack, shape, load, empty, save
from numpy.random import shuffle
import pandas as pd


# get hyperparameters for model
hyperp = get_hyperp()
# load generated GMM model Data
path = "Genr_Data/GMM/"
X = load(path + 'X.npy')
Y = load(path + 'Y.npy')
Z = load(path + 'Z.npy')
X_v = load(path + 'X_v.npy')
Y_v = load(path + 'Y_v.npy')
Z_v = load(path + 'Z_v.npy')

# shuffle the training data for sampling purpose
train = hstack((X,Y.reshape(len(Y),1)))
train = hstack((train, Z.reshape(len(Z),1)))
shuffle(train)

X = train[:,:10]
Y = train[:,10]
Z = train[:,11]

# generate a model for estimating the parameters of the
# true model based on the observations (X, Y) we just made
def model_fit(X,Y,X_v,Y_v):
    model = EM_algo_MM(hyperp, X, Y)
    i, logl_train, r = model.EM_fit()
    print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
            (logl_train, i, r))
    print("")
    print("MAP estimate of true model parameters:")
    model.print_map()
    print("")

    # # crossvalidate the estimated model with the validation data
    fit_params = model.get_p()
    model_v = EM_algo_MM(hyperp, X_v, Y_v)
    model_v.set_p(fit_params)
    logl_val, ll = model_v.logl()
    print("Crossvalidated logl: %.2f" % (logl_val))
    return logl_train, logl_val

d = range(1,11)
s = range(10,210,10)

likelihood_train = empty((len(d), len(s)))
likelihood_val = empty((len(d), len(s)))

for i in range(len(d)):
    for j in range(len(s)):
        dim = d[i];size = s[j]
        train_X = X[:size,:dim]
        train_Y = Y[:size]
        # train_Z = Z[:size]
        val_X = X_v[:,:dim]
        val_Y = Y_v
        logl_train, logl_val = model_fit(train_X, train_Y, val_X, val_Y)
        likelihood_train[i,j] = logl_train
        likelihood_val[i,j] = logl_val


for i in range(len(d)):
    for j in range(len(s)):
        print("Training Size:", s[j], "and Training Dimension", d[i])
        print("Training Likelihood:", int(likelihood_train[i,j]))
        print("Validation Likelihood:", int(likelihood_val[i,j]))

save('likelihood/Q4/GMM_GMM_train.npy', likelihood_train)
save('likelihood/Q4/GMM_GMM_val.npy', likelihood_val)
pd.DataFrame(likelihood_train).to_csv('likelihood/Q4/GMM_GMM_train.csv', header = None, index = None)
pd.DataFrame(likelihood_val).to_csv('likelihood/Q4/GMM_GMM_val.csv', header = None, index = None)

# print given dimension or training size
# example: print dimension = 6
print likelihood_val[d.index(6),:]



