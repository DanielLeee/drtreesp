import numpy as np
from scipy.optimize import OptimizeResult
import time

def adam(
    func_grad,
    x0,
    args=(),
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of ADAM -
    [http://arxiv.org/pdf/1412.6980.pdf].

    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    lr = learning_rate
    for i in range(startiter, startiter + maxiter):

        tot_time = 0
        t0 = time.time()
        f, g = func_grad(x)
        t1 = time.time()
        tot_time += t1 - t0
        
        
        print('adam iter {}, step = {:.8f}, obj = {}'.format(i, lr, f))

        if callback and callback(x):
            break

        t0 = time.time()

        m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
        v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
        mhat = m / (1 - beta1**(i + 1))  # bias correction.
        vhat = v / (1 - beta2**(i + 1))
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        t1 = time.time()
        tot_time += t1 - t0
        print('Time elapsed: {}'.format(tot_time))

    i += 1

    return OptimizeResult(x=x, fun=f, jac=g, nit=i, nfev=i, success=True)

