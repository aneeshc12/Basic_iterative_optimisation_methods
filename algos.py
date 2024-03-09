from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import copy

import os


# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:
    # Complete this function

    f_val = np.empty(int(1e4 + 5))
    f_prime_val = np.empty(int(1e4 + 5))

    f_val[0] = f(initial_point)
    f_prime_val[0] = np.linalg.norm(d_f(initial_point))

    ip = copy.copy(initial_point)

    if condition == "Backtracking":
        alpha = 10.
        rho = 0.75
        c = 0.001
        k = 0
        eps = 1e-6

        x_k = initial_point

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            d_k = -d_f(x_k)
            while f(x_k + alpha * d_k) > f(x_k) + c*alpha* d_f(x_k).T @ d_k:
                alpha *= rho
            
            x_k += alpha * d_k
            k += 1
        
            # log
            f_val[k] = f(x_k)
            f_prime_val[k] = np.linalg.norm(d_f(x_k))

        plot_iterations(k, f_val[:k], f_prime_val[:k], f.__name__, ip, condition)
        return x_k

    elif condition == "Bisection":
        c1 = 1e-3
        c2 = .1
        alpha = 0.
        t = 1.
        beta = 1e6
        k = 0
        eps = 1e-6

        cutoff = int(1e6)

        x_k = initial_point

        while k < 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            d_k = -d_f(x_k)
            while True:
                # print(ll, t, alpha, beta, x_k, d_k, np.linalg.norm(d_k))
                # print(f"\t, {f(x_k + t * d_k)} | {f(x_k) + c1 * t * (d_f(x_k).T @ d_k)}")
                # print(f"\t, {d_f(x_k + t * d_k).T @ d_k} | {c2 * (d_f(x_k).T @ d_k)}")

                if f(x_k + t * d_k) > f(x_k) + c1 * t * (d_f(x_k).T @ d_k):
                    beta = t 
                    t = (alpha + beta) / 2.
                elif d_f(x_k + t * d_k).T @ d_k < c2 * (d_f(x_k).T @ d_k):
                    alpha = t 
                    t = (alpha + beta) / 2.
                else:
                    break
            
                if alpha == beta:
                    # failed to converge
                    break

            x_k = x_k + t * d_k
            k += 1
        
            # log
            f_val[k] = f(x_k)
            f_prime_val[k] = np.linalg.norm(d_f(x_k))

        plot_iterations(k, f_val[:k], f_prime_val[:k], f.__name__, ip, condition)
        return x_k
    else:
        raise Exception

    # Use file f"plots/{f.__name__}_{np.array2string(initial_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(initial_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(initial_point)}_condition_cont.png" for plotting the contour plot
    pass


# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    f_val = np.empty(int(1e4))
    f_prime_val = np.empty(int(1e4))

    f_val[0] = f(initial_point)
    f_prime_val[0] = np.linalg.norm(d_f(initial_point))

    ip = copy.copy(initial_point)

    if condition == "Pure":
        x_k = initial_point
        k = 0
        eps = 1e-6

        while k <= 1e4:
            d_k = - np.linalg.inv(d2_f(x_k)) @ d_f(x_k)

            x_k = x_k + d_k
            k += 1

            # log
            f_val[k] = f(x_k)
            f_prime_val[k] = np.linalg.norm(d_f(x_k))

            if np.linalg.norm(d_f(x_k)) < eps:
                break   
        plot_iterations(k, f_val[:k], f_prime_val[:k], f.__name__, ip, condition)
        return x_k        

    elif condition == "Damped":
        alpha = 0.1
        beta = 0.9
        eps = 1e-6
        x_k = initial_point
        k = 0

        while k <= 1e4:
            d_k = -np.linalg.inv(d2_f(x_k)) @ d_f(x_k)

            t = 1
            while f(x_k) - f(x_k + t * d_k) < -alpha * t * d_f(x_k).T@d_k:
                t *= beta
            
            x_k = x_k + t * d_k
            k += 1

            # log
            f_val[k] = f(x_k)
            f_prime_val[k] = np.linalg.norm(d_f(x_k))

            if np.linalg.norm(d_f(x_k)) < eps:
                break
        plot_iterations(k, f_val[:k], f_prime_val[:k], f.__name__, ip, condition)
        return x_k

    elif condition == "Levenberg-Marquardt":
        x_k = initial_point
        k = 0
        eps = 1e-6

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            e_val, e_vec = np.linalg.eig(d2_f(x_k))
            l_min = np.min(e_val)

            if l_min <= 0:
                mu_k = -l_min + 0.1
                d_k = -np.linalg.inv(d2_f(x_k) + mu_k * np.eye(d_f(x_k).shape[0])) @ d_f(x_k)
            else:
                d_k = -np.linalg.inv(d2_f(x_k)) @ d_f(x_k)
            
            x_k = x_k + d_k
            k += 1

            # log
            f_val[k] = f(x_k)
            f_prime_val[k] = np.linalg.norm(d_f(x_k))

        plot_iterations(k, f_val[:k], f_prime_val[:k], f.__name__, ip, condition)
        return x_k
    
    elif condition == "Combined":
        x_k = initial_point
        k = 0
        c = 0.001
        eps = 1e-6
        alpha = 10.
        rho = 0.75

        while k <= 1e4 and np.linalg.norm(d_f(x_k)) > eps:
            e_val, e_vec = np.linalg.eig(d2_f(x_k))
            l_min = np.min(e_val)

            if l_min <= 0:
                mu_k = -l_min + 0.1
                d_k = -np.linalg.inv(d2_f(x_k) + mu_k * np.eye(x_k.shape[0])) @ d_f(x_k)
            else:
                d_k = -np.linalg.inv(d2_f(x_k)) @ d_f(x_k)
            
            while f(x_k + alpha * d_k) > f(x_k) + c*alpha* d_f(x_k).T @ d_k:
                alpha *= rho            

            x_k = x_k + alpha * d_k
            k += 1

            # log
            f_val[k] = f(x_k)
            f_prime_val[k] = np.linalg.norm(d_f(x_k))

        plot_iterations(k, f_val[:k], f_prime_val[:k], f.__name__, ip, condition)
        return x_k

    else:
        raise Exception


    # Use file f"plots/{f.__name__}_{np.array2string(initial_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(initial_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(initial_point)}_condition_cont.png" for plotting the contour plot
    pass

def plot_iterations(k, f_val, f_prime_val, fname, initial_point, condition):
    os.makedirs('./plots', exist_ok=True)

    val_path_name = f"plots/{fname}_{np.array2string(initial_point)}_{condition}_vals.png"
    grad_path_name = f"plots/{fname}_{np.array2string(initial_point)}_{condition}_grad.png"

    plt.figure()
    plt.plot([i for i in range(k)], f_val)
    plt.title(f"{fname} - {condition}: {np.array2string(initial_point)} vals")
    plt.savefig(val_path_name)
    plt.close()


    plt.figure()
    plt.plot([i for i in range(k)], f_prime_val)
    plt.title(f"{fname} - {condition}: {np.array2string(initial_point)} grads")
    plt.savefig(grad_path_name)
    plt.close()

def plot_contours():
    os.makedirs('./temp', exist_ok=True)
