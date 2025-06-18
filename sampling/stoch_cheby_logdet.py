import torch
import math
import time

def generate_spd_matrix(n, device='cpu'):
    A = torch.randn(n, n, device=device)
    A = A @ A.T
    A += n * torch.eye(n, device=device)
    return A

def power_iteration(A_mv, n, num_iter=20, device='cuda'):
    v = torch.randn(n, device=device)
    v /= v.norm()
    for _ in range(num_iter):
        v = A_mv(v)
        v /= v.norm()
    return v

def estimate_lambda_max(A_mv, n, num_iter=20, device='cuda'):
    v = power_iteration(A_mv, n, num_iter, device=device)
    Av = A_mv(v)
    return torch.dot(v, Av)

def estimate_lambda_min(A_mv, n, num_iter=20, device='cuda'):
    return torch.tensor(1e-3, device=device)

def chebyshev_coeff_log(k, a, b):
    #we approx log(x) as a weighted sum of chebyshev polynomials, with coeffs c 
    coeffs = []
    for j in range(k + 1): #each T_j(x) 
        s = 0.0
        for l in range(k + 1):
            x = math.cos(math.pi * (l + 0.5) / (k + 1))
            t = 0.5 * (b - a) * x + 0.5 * (b + a)
            s += math.log(t) * math.cos(math.pi * j * (l + 0.5) / (k + 1))
        c = (2 / (k + 1)) * s
        if j == 0:
            c /= 2
        coeffs.append(c)
    return torch.tensor(coeffs, dtype=torch.float32)

def stochastic_chebyshev_logdet(A_mv, n, k=20, num_samples=25, device='cuda'):
    lambda_max = estimate_lambda_max(A_mv, n, device=device)
    lambda_min = estimate_lambda_min(A_mv, n, device=device)
    a, b = lambda_min, lambda_max
    coeffs = chebyshev_coeff_log(k, a.item(), b.item()).to(device)

    total = torch.tensor(0.0, device=device)
    for _ in range(num_samples):
        v = (torch.randint(0, 2, (n,), dtype=torch.float32, device=device) * 2 - 1)
        v /= v.norm()

        Tkm2 = v
        Tkm1 = (2 / (b - a)) * (A_mv(v) - ((b + a) / 2) * v)
        trace_estimate = coeffs[0] * torch.dot(v, Tkm2) + coeffs[1] * torch.dot(v, Tkm1)

        for j in range(2, k + 1):
            T_k = (4 / (b - a)) * (A_mv(Tkm1) - ((b + a) / 2) * Tkm1) - Tkm2
            trace_estimate += coeffs[j] * torch.dot(v, T_k)
            Tkm2, Tkm1 = Tkm1, T_k

        total += trace_estimate

    return (n * total / num_samples).item() #multiply by n to undo the normalization, recover trace 

# main execution
n = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

A = generate_spd_matrix(n, device=device)
A_mv = lambda v: A @ v

_ = A @ torch.randn(n, device=device)
_ = torch.logdet(A)
torch.cuda.synchronize()

# stochastic Chebyshev timing
start_time = time.time()
approx_logdet = stochastic_chebyshev_logdet(A_mv, n, k=10, num_samples=50, device=device)
torch.cuda.synchronize()
cheby_time = time.time() - start_time

# exact logdet timing
start_time = time.time()
true_logdet = torch.logdet(A).item()
torch.cuda.synchronize()
true_time = time.time() - start_time

# print results 
rel_error = abs(approx_logdet - true_logdet) / abs(true_logdet)
print(f"\n=== Log-determinant Comparison ===")
print(f"Approximate log(det(A)): {approx_logdet:.4f}")
print(f"True log(det(A)):        {true_logdet:.4f}")
print(f"Relative error:          {rel_error:.4e}")
print(f"\nTime taken (Chebyshev):  {cheby_time:.4f} seconds")
print(f"Time taken (torch.logdet): {true_time:.4f} seconds")

