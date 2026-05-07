"""Analytic DCM fit functions"""
import numpy as np


def cavity_DCM(x, Q, Qc, w1, phi):
    """DCM fit function for notch-type resonator (transmission)."""
    return np.array(1 - Q / Qc * np.exp(1j * phi) / (1 + 1j * (x - w1) / w1 * 2 * Q))


def one_cavity_peak_abs(x, Q, Qc, w1):
    """Ideal resonator magnitude — used for initial guess fitting."""
    return np.abs(Q / Qc / (1 + 1j * (x - w1) / w1 * 2 * Q))

def one_cavity_peak(x, Q, Qc, w1):
    """Ideal resonator magnitude — used for initial guess fitting."""
    return np.abs(Q / Qc / (1 + 2 * 1j * Q * (x - w1) / w1))

def fit_raw_compare(x, y, params, method):
    """
    Compare fit to raw data. Returns normalised residual magnitude.
    Only DCM is supported; raises ValueError for any other method string.
    """
    if method == 'DCM':
        yfit = cavity_DCM(x, *params)
    else:
        raise ValueError(
            f"fit_raw_compare: unsupported method '{method}'. Only 'DCM' is supported."
        )
    return np.abs(y - yfit) / np.abs(y)

def min_one_Cavity_dip(parameter, x, data=None):
    """
    Least-squares residual for DCM fitting (lmfit minimiser target).

    Parameters
    ----------
    parameter : lmfit.Parameters
        Must contain keys: Q, Qc, w1, phi
    x : np.ndarray
        Frequency data.
    data : np.ndarray (complex)
        S21 data.
    """
    Q   = parameter['Q']
    Qc  = parameter['Qc']
    w1  = parameter['w1']
    phi = parameter['phi']

    model = cavity_DCM(x, Q, Qc, w1, phi)

    resid_re = model.real - data.real
    resid_im = model.imag - data.imag
    return np.concatenate((resid_re, resid_im))