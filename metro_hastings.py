import torch 
import math

#the distribution to sample from (known up to a constant)
def f_distr(x):
    #possible multivariate distributions
    #return torch.exp(-0.5 * torch.dot(x, x)) -> multivariate gaussian
    #return torch.exp(-torch.norm(x) ** 2)
    return torch.exp(-torch.norm(x/3))

class MultiGaussian: 
    #sophie's custom class
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        self.mu = mu
        self.cov = cov
        self.cov_inv = torch.linalg.inv(cov)
        self.dim = mu.shape[0]
        self.norm_const = torch.sqrt((2 * torch.pi) ** self.dim * torch.linalg.det(cov))
        self.dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)

    def pdf(self, x:torch.Tensor):
        density = torch.exp(-0.5*(x-mu).T @ self.cov_inv @ (x-mu))
        return density/self.norm_const

    def sample(self):
        return self.dist.sample()

#Cholesky decomposition is a computational lin. alg technique 
#given a covariance matrix C with Cholesky Decomp C = L@L.T
#let x = mu + Lz
#we sample z from the standard multi-variate Gaussian, N(0, I)
#this transformation allows us to sample x from N(mu, C)
def cholesky_decomp(X): 
    #must be a positive semidefinite matrix, entries must be REAL
    assert X.shape[0] == X.shape[1], "you need to provide a square matrix."
    n = X.shape[0]
    L = torch.zeros((n,n))

    #solve row by row (assuming i< j, then exploit symmetry)
    #so we first solve for col 1 of L (l_00, l_10...l_n0)
    for i in range(n):  
        for j in range(i, n):
            if i == j: #diagonal
                if i == 0: L[i,i] = torch.sqrt(X[0,0])
                else: L[i,i] = torch.sqrt(X[i,i] - X[i-1, i-1])
            else: #off diagonal
                #note: X_ij = known_sum + L[i,i]*L[j,i]
                known_sum = torch.dot(L[i, :i], L[j, :i]) #is indeed 0 when i = 0 
                L[j,i] = (X[i,j] - known_sum)/L[i,i]
    assert torch.allclose(L @ L.T, X), "i did it wrong then oops"
    return L

#example use case: 
X = torch.tensor([[2, 1, 0], [1, 3, 2], [0, 2, 4]], dtype=torch.float)
L = cholesky_decomp(X)

#using manually coded cholesky decomp
class AltMultiGaussian: 
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        self.mu = mu
        self.cov = cov
        self.dim = mu.shape[0]
        self.st_cov = torch.eye(len(mu)) #identity matrix 
        self.multi_standard_dist = torch.distributions.MultivariateNormal(torch.zeros(self.dim), self.st_cov) 
        self.L = cholesky_decomp(cov)

    def sample(self): 
        z = self.multi_standard_dist.sample()
        return self.mu + self.L @ z

#example use case: 
mu = torch.tensor([1.0, 2.0, 3.0])
cov = torch.tensor([
    [4.0, 2.0, 0.0],
    [2.0, 5.0, 1.0],
    [0.0, 1.0, 3.0]
])

chol_gaussian = AltMultiGaussian(mu, cov)

# Sample a point from N(mu, cov)
sample = chol_gaussian.sample()
print("Sample from N(mu, cov):", sample)

def prop(mu, cov = None, sigma="id"):
    #return torch.distributions.Normal(loc=mu, scale=sigma) single variable case 
    if sigma == "id": 
        sigma_val = 1.0
        cov = sigma_val**2 * torch.eye(len(mu))  # identity covariance
    if sigma == "prev_cov": 
        assert cov is not None, "you have to provide a covariance matrix when using 'prev_cov'."
        cov = cov

    prop_gaussian = MultiGaussian(mu, cov)
    return prop_gaussian

#fixed proposal distribution (i.e. gaussian)
def metro_hastings_fixed(n:int, f, prop, x_samp): 
    x_samp.append(torch.zeros(n))

    its = 5000
    for i in range(its): 
        #proposed sample
        prev = x_samp[-1]
        d = prop(mu = prev, sigma = "id")
        x_new = d.sample()

        #prob of proposing x_new given we're at prev
        pdf_new_given_old = d.pdf(torch.tensor(x_new))

        #prob of proposing prev given we're at x_new (reverse distribution)
        d_old_given_new = prop(mu = x_new, sigma = "id")
        pdf_old_given_new = d_old_given_new.pdf(torch.tensor(prev))

        p_accept = torch.min(torch.tensor(1.0), (f_distr(x_new)/f_distr(prev)) * (pdf_old_given_new / pdf_new_given_old))
        bern = torch.bernoulli(p_accept)

        if bern == 1: 
            x_samp.append(x_new)
        else: 
            x_samp.append(prev)

def get_cov(X):
    # each row is an observation, each col is a variable
    X = torch.stack(X)
    means = torch.mean(X, dim=0)
    X = X - means
    return torch.matmul(X.T, X) / (X.shape[0] - 1)

#adaptive gaussian mixture method
def MH_adaptive(n:int, f, prop, x_samp):
    x_samp.append(torch.zeros(n))

    its = 5000
    for i in range(its): 
        prev = x_samp[-1]

        #the key difference: we use the existing cov-matrix!!
        d = prop(mu = prev, cov = get_cov(x_samp), sigma = "prev_cov")
        x_new = d.sample()

        pdf_new_given_old = d.pdf(torch.tensor(x_new))
        d_old_given_new = prop(mu = x_new, cov = get_cov(x_samp), sigma = "prev_cov")
        pdf_old_given_new = d_old_given_new.pdf(torch.tensor(prev))

        p_accept = torch.min(torch.tensor(1.0), (f_distr(x_new)/f_distr(prev)) * (pdf_old_given_new / pdf_new_given_old))
        bern = torch.bernoulli(p_accept)
        if bern == 1: x_samp.append(x_new)
        else: x_samp.append(prev)

x_samp = []
n_dim = 2  
#metro_hastings_fixed(n_dim, f_distr, prop, x_samp)

#works WAYYY Better (try the visualization)
MH_adaptive(n_dim, f_distr, prop, x_samp) 

burn_in = 500
samples = torch.stack(x_samp[burn_in:])
import matplotlib.pyplot as plt

if n_dim == 2:
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5)
    plt.title("2D Metro-Hastings Samples")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.grid(True)
    plt.axis('equal')
    plt.show()


