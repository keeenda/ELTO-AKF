from itertools import combinations
import numpy as np
import scipy.io as sio


def print_pde(
    pde_coef, rhs_description, threshold=0, n_print=None, sort=False, ut="u_t"
):
    pde_coef = np.array(pde_coef).ravel()
    rhs_description = np.array(rhs_description).ravel()
    assert len(pde_coef) == len(rhs_description)
    if sort:
        size_ranking = np.argsort(np.abs(pde_coef.ravel()))[::-1]
        rhs_description = rhs_description[size_ranking]
        w = pde_coef[size_ranking]
    else:
        w = pde_coef
    if n_print is None:
        n_print = len(w)
    n_print = min(n_print, len(w))
    pde = ut + " = "
    first = True
    for i in range(n_print):
        if abs(w[i]) > threshold:
            if not first:
                pde = pde + " + "
            pde = (
                pde
                + "(%05f %+05fi)" % (w[i].real, w[i].imag)
                + rhs_description[i]
                + "\n   "
            )
            first = False
    print(pde)
    return pde


def read_2d_data(file_path):
    data = sio.loadmat(file_path)
    u = None
    for _ in ["usol", "u", "uu"]:
        if _ in data:
            u = data[_]
            break
    if u is None:
        raise AttributeError(
            "State variable 'usol', 'u', or 'uu' is missing from the file."
        )
    else:
        u = u.real
        try:
            du = data["du"]
        except KeyError:
            du = None
        # if 'du' is not in data:
        #     du = None
        # else:
        #     du = data["du"]
        x = data["x"]
        t = data["t"]
        return u, du, x, t


def add_noise(u, noise_lv):
    return u + 0.01 * abs(noise_lv) * np.std(u) * np.random.randn(*u.shape)


def best_subset(X, y, p):
    """
    Brute-force best subset selection using numpy.linalg.lstsq

    Parameters:
    X (np.ndarray): Feature matrix (n_samples, n_features)
    y (np.ndarray): Response vector (n_samples,)
    p (int): Number of active terms (features) to select

    Returns:
    best_indices (tuple): Indices of the best p predictors
    best_coef (np.ndarray): Coefficients of the best model (p,)
    min_rss (float): Residual sum of squares of the best model
    """
    n_samples, n_features = X.shape
    min_rss = np.inf
    best_indices = None
    best_coef = None

    for subset in combinations(range(n_features), p):
        X_subset = X[:, subset]
        coef, residuals, _, _ = np.linalg.lstsq(X_subset, y, rcond=None)

        if residuals.size > 0:
            rss = residuals[0]
        else:
            # If underdetermined system, compute RSS manually
            y_pred = X_subset @ coef
            rss = np.sum((y - y_pred) ** 2)

        if rss < min_rss:
            min_rss = rss
            best_indices = subset
            best_coef = coef

    return best_indices, best_coef, min_rss
