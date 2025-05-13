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
        #assumes x has shape 1 x d 
        if x.dim() == 1:
            x = x.unsqueeze(0)
        density = torch.exp(-0.5*(x-self.mu) @ self.cov_inv @ (x-self.mu).T)
        return density/self.norm_const

    def sample(self):
        return self.dist.sample()

#Cholesky decomposition is a computational lin. alg technique 
#given a covariance matrix C, decomposed as C = L@L.T, where L lower triangular
#fact: any positive semidefinite matix is guaranteed a unique Cholesky Decomp
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
#let x = mu + Lz
#we sample z from the standard multi-variate Gaussian, N(0, I)
#this transformation allows us to sample x from N(mu, C)
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

from torch.distributions import Categorical
class GaussianMixture: 
    def __init__(self, mix_w:torch.Tensor, mu: torch.Tensor, cov: torch.Tensor):
        self.mix_w = mix_w
        self.categorical = Categorical(mix_w)
        self.mu = um
        self.cov = cov
        self.N_gauss = mu.shape[0]
        #the individ gaussians
        self.components = [ MultiGaussian(mu[i], cov[i]) for i in range(self.N_gauss)]

    def pdf(self, x):
        tot = 0
        for i in range(self.N_gauss):
            tot += self.mix_w[i] * self.components[i].pdf(x)
        return tot

    def sample(): 
        indx = self.categorical.sample() #which Gaussian to sample from 
        return self.components[indx].sample()

#Kronecker Product of two matrices
def kron_prod(X, Y):
    m,n = X.shape[0], X.shape[1]
    prod = []
    for i in range(m):
        kron_row_i = []
        for j in range(n):
            kron_row_i.append(X[i,j]*Y)
        kron_row_i = torch.cat(kron_row_i, dim = 1)
        prod.append(kron_row_i)

    prod = torch.cat(prod, dim = 0)
    return prod

#more complex: we have training period, updating weight parameters of prop distr's gaussian
#most resembles ML
#paper: https://arxiv.org/pdf/1212.0122
def MH_mixture_adaptive(d, f, x_samp, T_train = 500, T_stop = 500, T_tot = 5000, N_gauss = 2):
    # init of gaussian mixture parameters
    #mu = torch.randn((N_gauss, d))
    mu = torch.full((N_gauss, d), torch.distributions.Uniform(-4, 4).sample())
    S = [mu[i].unsqueeze(0) for i in range(N_gauss)]
    covs = torch.zeros((N_gauss, d, d))
    mix_w = torch.full((N_gauss,), 1/float(N_gauss))

    #emprically, random cov init works POORLY while the identity works well
    # Possibility 1: random Cov init
    for i in range(N_gauss):
        random_matrix = torch.rand(d, d)
        cov_matrix = random_matrix @ random_matrix.T  # ensure symmetry
        covs[i] = cov_matrix

    # Possibility 2: Identity matrix, large variance for exploration
    for i in range(N_gauss):
        covs[i] = 10*torch.eye(d)

    for i in range(T_tot):
        prev = x_samp[-1]
        d_prop = GaussianMixture(mix_w, mu, covs)
        x_new = d_prop.sample()  # shape should be (1, d)

        # Compute acceptance probability
        pdf_new = d_prop.pdf(x_new.clone().detach())
        pdf_old = d_prop.pdf(prev.clone().detach())
        p_accept = torch.min(torch.tensor(1.0), (f(x_new)/f(prev)) * (pdf_old / pdf_new))
        bern = torch.bernoulli(p_accept)
        if bern == 1:
            x_samp.append(x_new)
        else:
            x_samp.append(prev)

        # Update proposal parameters
        if i < T_stop:
            t_dis = torch.sum((mu - x_new) ** 2, dim=1)
            j = torch.argmin(t_dis)

            if x_new.dim() == 1: x_new = x_new.unsqueeze(0)
            S[j] = torch.cat((S[j], x_new), dim=0)  # row-wise

            if i > T_train:
                m_j = S[j].shape[0]
                mu[j] = ((m_j-1)/m_j)*mu[j] + (1/m_j)*x_new.squeeze(0)  # update mean
                S_tilde = S[j] - mu[j]
                EPS = 1e-4
                covs[j] = ((S_tilde.T @ S_tilde) + (m_j - 1)*EPS*torch.eye(d))/(m_j - 1)

                # Update weights
                ents = [S[i].shape[0] for i in range(N_gauss)]
                tot_ents = sum(ents)
                for i in range(N_gauss):
                    mix_w[i] = ents[i] / tot_ents

####################################
##### TESTING ######################

x_samp = []
n_dim = 2  
#metro_hastings_fixed(n_dim, f_distr, prop, x_samp)

##### Adaptive Model -> works WAYYY Better (try the visualization)
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


###### Adaptive Gaussian MIXTURE model 
def target_density(x):
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return torch.exp(-((x**2 - 4)**2)/4).squeeze()

torch.manual_seed(42)
N = 2
mu_init = torch.tensor([
    np.random.uniform(-4, 0),
    np.random.uniform(0, 4)
], dtype=torch.float32)

covs_init = torch.tensor([10.0, 10.0])
weights_init = torch.tensor([0.5, 0.5])

# Initial sample point x0 ~ N(0, 1)
x0 = torch.normal(0.0, 1.0, size=(1,))  # shape: [1]
x_samp = [x0]

MH_mixture_adaptive(
    d=1,
    f=target_density,
    x_samp=x_samp,
    T_train=0,
    T_stop=0,
    T_tot=5000,
    N_gauss=2)

samples = torch.stack(x_samp, dim=0)
# Histogram
plt.hist(samples, bins=100, density=True, alpha=0.6, label='MH samples')
xs = torch.linspace(-4, 4, 1000)
with torch.no_grad():
    ys = target_density(xs)
    ys = ys / torch.trapz(ys, xs)  
plt.plot(xs.numpy(), ys.numpy(), 'r--', label='Target density')
plt.title("AGM-MH Sampling: Bimodal Univariate Example")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

#Variations to test: 
#How initialization affects the sampling
#Train, Stop periods 
#Why random covaraince init works worse than identity 
    #how large vs small variance affects it
# N_gaussians parameter 

#Test performance of 3 sampling methods 
