import torch 
import math

#step 1: fixed proposal distribution (i.e. gaussian)
def f_distr(x):
    #possible multivariate distributions
    #return torch.exp(-0.5 * torch.dot(x, x)) -> multivariate gaussian
    #return torch.exp(-torch.norm(x) ** 2)
    return torch.exp(-torch.norm(x/3))


def prop(mu, sigma= 0.5):
    #return torch.distributions.Normal(loc=mu, scale=sigma) single variable case 
    cov = sigma**2 * torch.eye(len(mu))  # identity covariance
    return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)


def metro_hastings_fixed(n:int, f, prop, x_samp): 
    x_samp.append(torch.zeros(n))

    its = 5000
    for i in range(its): 
        #proposed sample
        prev = x_samp[-1]
        d = prop(mu = prev)
        x_new = d.sample()
        

        d_old_given_new = prop(mu = x_new)
        #prob of proposing x_new given we're at prev
        pdf_new_given_old = torch.exp(d.log_prob(torch.tensor(x_new)))
        #prob of proposing prev given we're at x_new
        pdf_old_given_new = torch.exp(d.log_prob(torch.tensor(prev)))

        p_accept = torch.min(torch.tensor(1.0), (f_distr(x_new)/f_distr(prev)) * (pdf_old_given_new / pdf_new_given_old))
        bern = torch.bernoulli(p_accept)

        if bern == 1: 
            x_samp.append(x_new)
        else: 
            x_samp.append(prev)


x_samp = []
n_dim = 2  # try 2D for visualization
metro_hastings_fixed(n_dim, f_distr, prop, x_samp)

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



