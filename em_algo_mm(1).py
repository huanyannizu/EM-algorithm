# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: xinyue.li@aalto.fi, 2016

from numpy import outer, eye, ones, zeros, diag, log, sqrt, exp, pi, vstack, hstack, argmax
from numpy.linalg import inv, solve, det
from numpy.random import multivariate_normal as mvnormal, normal, gamma, beta, binomial
from scipy.special import gammaln
import math

from em_algo import EM_algo

class EM_algo_MM(EM_algo):
    """
        A mixture of two linear models.
    """

    def reset(self):
        """
            Reset priors and draw parameter estimates from prior.
        """
        # priors
        self.lbd_phi0       = self.h["lbd_phi0"] #lambda0 = 1
        self.alpha_s20      = self.h["alpha_s20"] #alpha(sigma^2) = 5
        self.beta_s20       = self.h["beta_s20"] #beta(sigma^2) = 1
        self.sigma_phi0     = eye(self.pdata) * self.h["lbd_phi0"] #covariate matrix of sigma square
        self.sigma_phi0_inv = eye(self.pdata) / self.h["lbd_phi0"] #inverse
        self.mu_phi0        = ones(self.pdata) * self.h["mu_phi0"] #0
        self.alpha_w0       = self.h["alpha_w0"]
        self.beta_w0        = self.h["beta_w0"]

        # initial parameter estimates drawn from prior
        self.p = dict()
        self.p["sigma_square_1"] = 1.0 / gamma(self.alpha_s20, 1.0 / self.beta_s20)
        self.p["sigma_square_2"] = 1.0 / gamma(self.alpha_s20, 1.0 / self.beta_s20) 
        self.p["phi_1"] = mvnormal(self.mu_phi0, self.p["sigma_square_1"] * self.sigma_phi0)
        self.p["phi_2"] = mvnormal(self.mu_phi0, self.p["sigma_square_2"] * self.sigma_phi0)
        self.p["w"] = beta(self.alpha_w0, self.beta_w0)
        self.p["z"] = binomial(1, self.p["w"], self.ndata)

    def draw(self, item):
        """
            Draw a data sample from the current predictive distribution.
            Returns the y-value (and a constant z-value for compatibility)
        """
        l = len(item)
        mean = float(item.dot(self.p["phi_1"]))
        std  = sqrt(self.p["sigma_square_1"])
        return normal(mean, std), 1

    def GausProb(self, y, x, phi, s2):
        t1 = 1./sqrt(2*pi*s2)
        t2 = exp(-(sum(x * phi) - y)**2 / (2*s2))
        return t1*t2

    def logl(self):
        """
            Calculate the complete log likelihood for this model.
        """
        ll=zeros(6)
        l = len(self.Y)
        likelihood = 0

        # term1 logp(y)
        for i in range(l):
            p_y_1 = self.p["w"] * self.GausProb(self.Y[i], self.X[i], self.p["phi_1"], self.p["sigma_square_1"])
            p_y_2= (1-self.p["w"]) * self.GausProb(self.Y[i], self.X[i], self.p["phi_2"], self.p["sigma_square_2"])
            likelihood += log(p_y_1 + p_y_2)
        ll[0] = term1 = likelihood

        # term2 logp(phi_1)
        t21 = -0.5 * self.pdata * log(2*pi*self.p["sigma_square_1"])
        t22 = -0.5 * log(det(self.sigma_phi0))
        t23_1 = (self.p["phi_1"] - self.mu_phi0).T.dot(self.sigma_phi0_inv)
        t23_2 = t23_1.dot(self.p["phi_1"] - self.mu_phi0)
        t23 = -0.5 * t23_2 / self.p["sigma_square_1"]
        ll[1] = term2 = t21+t22+t23

        #term3 logp(phi_2)
        t31 = -0.5 * self.pdata * log(2*pi*self.p["sigma_square_2"])
        t32 = -0.5 * log(det(self.sigma_phi0))
        t33_1 = (self.p["phi_2"] - self.mu_phi0).T.dot(self.sigma_phi0_inv)
        t33_2 = t33_1.dot(self.p["phi_2"] - self.mu_phi0)
        t33 = -0.5 * t33_2 / self.p["sigma_square_2"]
        ll[2] = term3 = t31+t32+t33

        #term4 logp(s2_1)
        t41 = self.alpha_s20 * log(self.beta_s20)
        t42 = -log(math.gamma(self.alpha_s20))
        t43 = -(self.alpha_s20 + 1) * log(self.p["sigma_square_1"])
        t44 = -self.beta_s20 / self.p["sigma_square_1"]
        ll[3] = term4 = t41+t42+t43+t44

        #term5 logp(s2_2)
        t51 = self.alpha_s20 * log(self.beta_s20)
        t52 = -log(math.gamma(self.alpha_s20))
        t53 = -(self.alpha_s20 + 1) * log(self.p["sigma_square_2"])
        t54 = -self.beta_s20 / self.p["sigma_square_2"]
        ll[4] = term5 = t51+t52+t53+t54

        #term6 logp(w)
        t61 = log(math.gamma(self.alpha_w0 + self.beta_w0))
        t62 = -(log(math.gamma(self.alpha_w0) * math.gamma(self.beta_w0)))
        t63 = (self.alpha_w0 - 1) * log(self.p["w"])
        t64 = (self.beta_w0 - 1) * log(1 - self.p["w"])
        ll[5] = term6 = t61+t62+t63+t64


        return sum(ll), ll

    def gamma_z_update(self):
        prob_one = (1./sqrt(2*pi*self.p["sigma_square_1"])) * exp( -1./ (2*self.p["sigma_square_1"]) * (self.Y-self.X.dot(self.p["phi_1"]))**2)
        prob_two = (1./sqrt(2*pi*self.p["sigma_square_2"])) * exp( -1./ (2*self.p["sigma_square_2"]) * (self.Y-self.X.dot(self.p["phi_2"]))**2)
        self.gamma_z_one = self.p["w"] * prob_one / (self.p["w"] * prob_one + (1-self.p["w"]) * prob_two)
        self.gamma_z_two = 1 - self.gamma_z_one

    def EM_iter(self):
        """
            Executes a single round of EM updates for GMM model.

            Has checks to make sure that updates increase logl and
            that parameter values stay in sensible limits.
        """
        self.gamma_z_update()

        # phi one
        gamma_X = self.X * self.gamma_z_one.reshape(len(self.gamma_z_one), 1)
        gamma_XX = self.X.T.dot(gamma_X)
        term_one = gamma_XX + self.sigma_phi0_inv
        gamma_YX = self.X.T.dot(self.gamma_z_one * self.Y)
        sigma_inv_phi = self.sigma_phi0_inv.dot(self.mu_phi0)
        term_two = gamma_YX + sigma_inv_phi
        self.p["phi_1"] = inv(term_one).dot(term_two)


        #phi two
        gamma_X = self.X * self.gamma_z_two.reshape(len(self.gamma_z_two), 1)
        gamma_XX = self.X.T.dot(gamma_X)
        term_one = gamma_XX + self.sigma_phi0_inv
        gamma_YX = self.X.T.dot(self.gamma_z_two * self.Y)
        sigma_inv_phi = self.sigma_phi0_inv.dot(self.mu_phi0)
        term_two = gamma_YX + sigma_inv_phi
        self.p["phi_2"] = inv(term_one).dot(term_two)

        self.assert_logl_increased("phi update")

        #sigma2 one
        self.gamma_z_update()
        X_phi = self.X.dot(self.p['phi_1'])
        sigma_phi_1 = (self.p["phi_1"] - self.mu_phi0).T.dot(self.sigma_phi0_inv.dot(self.p["phi_1"] - self.mu_phi0))
        term_one = sum(self.gamma_z_one * (self.Y - X_phi)**2) + 2 * self.beta_s20 + sigma_phi_1
        term_two = sum(self.gamma_z_one) + 2 * self.alpha_s20 + 2 + self.pdata
        self.p["sigma_square_1"] = term_one / term_two
        if self.p["sigma_square_1"] < 0.0:
            raise ValueError("sigma_square_1 < 0.0")

        #sigma2 two
        X_phi = self.X.dot(self.p['phi_2'])
        sigma_phi_2 = (self.p["phi_2"] - self.mu_phi0).T.dot(self.sigma_phi0_inv.dot(self.p["phi_2"] - self.mu_phi0))
        term_one = sum(self.gamma_z_two * (self.Y - X_phi)**2) + 2 * self.beta_s20 + sigma_phi_2
        term_two = sum(self.gamma_z_two) + 2 * self.alpha_s20 + 2 + self.pdata
        self.p["sigma_square_2"] = term_one / term_two
        if self.p["sigma_square_2"] < 0.0:
            raise ValueError("sigma_square_2 < 0.0")  

        self.assert_logl_increased("sigma2 update") 

        #w
        self.gamma_z_update()
        self.p["w"] = (sum(self.gamma_z_one) + self.alpha_w0 - 1) / (len(self.Y) + self.alpha_w0 + self.beta_w0 - 2)
        
        self.assert_logl_increased("w update")

        #z
        self.gamma_z_update()
        prob = hstack((self.gamma_z_one.reshape(self.ndata,1), self.gamma_z_two.reshape(self.ndata,1)))
        self.p["z"] = argmax(prob, axis = 1)

    def print_p(self):
        """
            Prints the model parameters, one at each line.
        """
        print("phi    : %s" % (self.pretty_vector(self.p["phi_1"])))
        print("sigma2 : %.3f" % (self.p["sigma_square_1"]))

    def print_map(self):
        """
            Prints the model parameters, one at each line.
        """
        print("phi_1    : %s" % (self.pretty_vector(self.p["phi_1"])))
        print("sigma2_1 : %.3f" % (self.p["sigma_square_1"]))
        print("phi_2    : %s" % (self.pretty_vector(self.p["phi_2"])))
        print("sigma2_2 : %.3f" % (self.p["sigma_square_2"]))
        print("w        : %.3f" % (self.p["w"]))


