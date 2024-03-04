from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


# Do not rename or delete this function
def steepest_descent(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Backtracking", "Bisection"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
    pass


# Do not rename or delete this function
def newton_method(
    f: Callable[[npt.NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    d2_f: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    inital_point: npt.NDArray[np.float64],
    condition: Literal["Pure", "Damped", "Levenberg-Marquardt", "Combined"],
) -> npt.NDArray[np.float64]:
    # Complete this function
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_vals.png" for plotting f(x) vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_grad.png" for plotting |f'(x)| vs iters
    # Use file f"plots/{f.__name__}_{np.array2string(inital_point)}_condition_cont.png" for plotting the contour plot
    pass
