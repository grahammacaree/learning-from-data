import numpy as np

def vc_bound(N, dvc=50, delta=0.05):
    return (np.sqrt(8/N * np.log(4*(2*N)**dvc/delta)))

def rademacher_bound(N, dvc=50, delta=0.05):
    return (np.sqrt((2*(np.log(2*N) + dvc*np.log(N)))/N) + np.sqrt(2/N*np.log(1/delta)) + 1/N)

def parrondo_bound(N, dvc=50, delta=0.05):
    log_term = np.log(6/delta) + dvc * np.log(2*N)
    # solve eps^2 - (2/N)eps - (1/N)*log_term = 0
    a, b, c = 1, -2/N, -log_term/N
    return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

def devroye_bound(N, dvc=50, delta=0.05):
    log_term = np.log(4/delta) + dvc * np.log(N**2)
    a, b, c = 1 - 2/N, -2/N, -log_term/(2*N)
    return (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

print("vc: "+ str(vc_bound(5)))
print("rademacher: "+ str(rademacher_bound(5)))
print("parrondo: "+ str(parrondo_bound(5)))
print("devroye: "+ str(devroye_bound(5)))