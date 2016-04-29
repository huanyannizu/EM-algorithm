from em_algo_mm import EM_algo_MM
from em_algo_lm import EM_algo_LM
from generator import get_hyperp

from numpy import arange, min, max, sqrt, mean, std, hstack, vstack, shape, load, empty, save
from numpy.random import shuffle
import pandas as pd

NUM_EXP = 10

# get hyperparameters for model
hyperp = get_hyperp()
# load generated GMM model Data
path = "/Users/lixinyue/Documents/Machine_Learning_Advanced_Probabilistic_Methods/example_code_python_change/"
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
def GMM_fit(X,Y,X_v,Y_v):
    model = EM_algo_MM(hyperp, X, Y)
    i, logl_train, r = model.EM_fit()
    # print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
    #         (logl_train, i, r))
    # print("")
    # print("MAP estimate of true model parameters:")
    # model.print_map()
    # print("")

    # # crossvalidate the estimated model with the validation data
    fit_params = model.get_p()
    model_v = EM_algo_MM(hyperp, X_v, Y_v)
    model_v.set_p(fit_params)
    logl_val, ll = model_v.logl()
    # print("Crossvalidated logl: %.2f" % (logl_val))
    return logl_train, logl_val

def LM_fit(X,Y,X_v,Y_v):
    model = EM_algo_LM(hyperp, X, Y)
    i, logl_train, r = model.EM_fit()
    # print("Model fit (logl %.2f) after %d iterations (%s reached)" % \
    #         (logl_train, i, r))
    # print("")
    # print("MAP estimate of true model parameters:")
    # model.print_p()
    # print("")

    # # crossvalidate the estimated model with the validation data
    fit_params = model.get_p()
    model_v = EM_algo_LM(hyperp, X_v, Y_v)
    model_v.set_p(fit_params)
    logl_val, ll = model_v.logl()
    # print("Crossvalidated logl: %.2f" % (logl_val))
    return logl_train, logl_val

d = range(1,11)
s = range(10,210,10)

likelihood_train_lm = empty((len(d), len(s)))
likelihood_train_mm = empty((len(d), len(s)))

likelihood_val_lm = empty((len(d), len(s)))
likelihood_val_mm = empty((len(d), len(s)))

for i in range(len(d)):
    for j in range(len(s)):
        print i+j
        dim = d[i];size = s[j]
        train_X = X[:size,:dim]
        train_Y = Y[:size]
        # train_Z = Z[:size]
        val_X = X_v[:,:dim]
        val_Y = Y_v

        lt1 = 0.; lv1 = 0.; lt2 = 0.; lv2 = 0.; 
        for k in range(NUM_EXP):
            # train with GMM
            logl_train, logl_val = GMM_fit(train_X, train_Y, val_X, val_Y)
            lt1 += logl_train; lv1 += logl_val

            # train with LM
            logl_train, logl_val = LM_fit(train_X, train_Y, val_X, val_Y)
            lt2 += logl_train; lv2 += logl_val

        likelihood_train_mm[i,j] = lt1 / NUM_EXP
        likelihood_val_mm[i,j] = lv1 / NUM_EXP
        likelihood_train_lm[i,j] = lt2 / NUM_EXP
        likelihood_val_lm[i,j] = lv2 / NUM_EXP

# save('likelihood/Q5/b/GMM_GMM_train.npy', likelihood_train_mm)
# save('likelihood/Q5/b/GMM_GMM_val.npy', likelihood_val_mm)
# save('likelihood/Q5/b/GMM_LM_train.npy', likelihood_train_lm)
# save('likelihood/Q5/b/GMM_LM_val.npy', likelihood_val_lm)

pd.DataFrame(likelihood_train_mm).to_csv('/Users/lixinyue/Documents/Machine_Learning_Advanced_Probabilistic_Methods/example_code_python_change/GMM_GMM_train_average.csv', header = None, index = None)
pd.DataFrame(likelihood_val_mm).to_csv('/Users/lixinyue/Documents/Machine_Learning_Advanced_Probabilistic_Methods/example_code_python_change/GMM_GMM_val_average.csv', header = None, index = None)
pd.DataFrame(likelihood_train_lm).to_csv('/Users/lixinyue/Documents/Machine_Learning_Advanced_Probabilistic_Methods/example_code_python_change/GMM_LM_train_average.csv', header = None, index = None)
pd.DataFrame(likelihood_val_lm).to_csv('/Users/lixinyue/Documents/Machine_Learning_Advanced_Probabilistic_Methods/example_code_python_change/GMM_LM_val_average.csv', header = None, index = None)

# # print given dimension or training size
# # example: print dimension = 6
# print likelihood_val[d.index(6),:]



