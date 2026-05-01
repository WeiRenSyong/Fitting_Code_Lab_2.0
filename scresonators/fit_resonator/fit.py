import logging
import os
import sys

import inflect
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from uncertainties import ufloat, umath

import fit_resonator.cavity_functions as ff
import fit_resonator.plot as fp
import fit_resonator.resonator as res

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

np.set_printoptions(precision=4, suppress=True)
p = inflect.engine()


def extract_near_res(
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        f_res: float,
        kappa: float,
        extract_factor: int = 1):
    """Extract a portion of the spectrum around resonance."""
    xstart = f_res - extract_factor / 2 * kappa
    xend   = f_res + extract_factor / 2 * kappa

    mask   = (x_raw > xstart) & (x_raw < xend)
    x_temp = x_raw[mask]
    y_temp = y_raw[mask]

    if len(x_temp) < 1:
        raise ValueError("Failed to extract data from designated bandwidth")

    return x_temp, y_temp

def find_circle(
        x, y):
    """Least-squares circle fit (Kåsa method)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xavg = np.mean(x)
    yavg = np.mean(y)
    N    = len(x)

    xnew = x - xavg
    ynew = y - yavg

    Suu  = np.sum(xnew ** 2)
    Svv  = np.sum(ynew ** 2)
    Suv  = np.sum(xnew * ynew)
    Suuu = np.sum(xnew ** 3)
    Svvv = np.sum(ynew ** 3)
    Suvv = np.sum(xnew * ynew ** 2)
    Svuu = np.sum(ynew * xnew ** 2)

    Suv2    = Suv
    matrix1 = 0.5 * (Suuu + Suvv)
    matrix2 = 0.5 * (Svvv + Svuu)

    Suv     = Suv / Suu
    matrix1 = matrix1 / Suu
    Svv     = Svv - Suv * Suv2
    matrix2 = matrix2 - Suv2 * matrix1
    matrix2 = matrix2 / Svv
    matrix1 = matrix1 - Suv * matrix2

    alpha = matrix1 ** 2 + matrix2 ** 2 + (Suu + Svv) / N
    R     = alpha ** 0.5

    return matrix1 + xavg, matrix2 + yavg, R

def find_initial_guess(
        x, 
        y1, # real part
        y2, # imaginary part
        Method, 
        output_path, 
        plot_extra):
    """
    Determine initial guess for DCM parameters: [Q, Qc, f_c, phi]
    """
    if Method.method != "DCM":
        raise ValueError("It currently supports DCM only.")

    try:
        y  = y1 + 1j * y2
        y1 = np.real(y)
        y2 = np.imag(y)
    except Exception as e:
        raise ValueError(f"Problem initializing data in find_initial_guess(): {e}")

    try:
        x_c, y_c, r = find_circle(y1, y2)
        z_c = x_c + 1j * y_c
    except Exception as e:
        raise ValueError(f"Problem in find_circle(): {e}")

    try:
        ydata = y - 1
        z_c   = z_c - 1
    except Exception as e:
        raise ValueError(f"Error shifting data into canonical position: {e}")

    try:
        phi      = np.angle(-z_c) # phi is postive when z_c is at the third quadrant
        ydata    = ydata * np.exp(-1j * phi)
        freq_idx = np.argmin(np.abs(ydata))
        f_c      = x[freq_idx]
        z_c = z_c * np.exp(-1j * phi)

        Q_over_Qc = np.max(np.abs(ydata))
        y_temp    = np.abs(np.abs(ydata) - np.max(np.abs(ydata)) / np.sqrt(2))

        _, idx1 = find_nearest(y_temp[0:freq_idx], 0)
        _, idx2 = find_nearest(y_temp[freq_idx:], 0)
        idx2    = idx2 + freq_idx - 1

        kappa = abs(x[idx1] - x[idx2])   # Estimated linewidth
        Q     = f_c / kappa
        Qc    = Q / Q_over_Qc

        popt, _ = spopt.curve_fit(
            ff.one_cavity_peak_abs,
            x,
            np.abs(ydata),
            p0=[Q, Qc, f_c],
            bounds=([1e1, 1e1, 4e9], [1e9, 1e9, 8e9]) # bounds=( [min_Q, min_Qc, min_fc], [max_Q, max_Qc, max_fc] )
        )
        Q, Qc = popt[0], popt[1]
        f_c = popt[2]
        init_guess = [Q, Qc, f_c, phi]

        print(f'Initial guess shows that fc = {f_c/1e9:.6f} GHz.')
        print(f'Initial guess shows that linewidth = {kappa/1e3:.3f} kHz.')
        print(f'Initial guess shows that Q = {Q:.0f}.')
        print(f'Initial guess shows that Qc = {Qc:.0f}.')      

    except Exception as e:
        print(e)
        raise RuntimeError("Failed to find initial guess for DCM. Please manually initialize a guess.")

    return init_guess, x_c, y_c, r

def find_nearest(
        array, 
        value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def monte_carlo_fit(
        xdata=None, ydata=None, parameter=None, Method=None):
    """Monte Carlo refinement of DCM fit parameters."""
    assert xdata     is not None, "xdata is not defined"
    assert ydata     is not None, "ydata is not defined"
    assert parameter is not None, "parameter is not defined"
    assert Method    is not None, "Method is not defined"

    try:
        ydata_1stfit = Method.func(xdata, *parameter)

        weight_array = 1 / np.abs(ydata) if Method.MC_weight == 'yes' \
                       else np.ones(len(xdata))

        weighted_ydata       = weight_array * ydata
        weighted_ydata_1stfit = weight_array * ydata_1stfit
        error   = np.linalg.norm(weighted_ydata - weighted_ydata_1stfit) / len(xdata)
        error_0 = error
    except Exception as e:
        raise ValueError(f"Failed to initialize monte_carlo_fit(): {e}")

    try:
        counts = 0
        while counts < Method.MC_rounds:
            counts += 1
            random = Method.MC_step_const * (np.random.random_sample(len(parameter)) - 0.5)

            # Zero out steps for fixed parameters
            if 'Q'   in Method.MC_fix: random[0] = 0
            if 'Qc'  in Method.MC_fix: random[1] = 0
            if 'w1'  in Method.MC_fix: random[2] = 0
            if 'phi' in Method.MC_fix: random[3] = 0

            random[3] = random[3] * 0.1     # smaller step for phi
            random    = np.exp(random)
            new_parameter    = parameter * random
            new_parameter[3] = np.mod(new_parameter[3], 2 * np.pi)

            ydata_MC         = Method.func(xdata, *new_parameter)
            weighted_ydata_MC = weight_array * ydata_MC
            new_error = np.linalg.norm(weighted_ydata_MC - weighted_ydata) / len(xdata)

            if new_error < error:
                parameter = new_parameter
                error     = new_error
    except Exception as e:
        raise RuntimeError(f"Error in monte_carlo_fit loop: {e}")

    if error < error_0:
        stop_MC = False
        print('Monte Carlo fit found better parameters.')
        if Method.manual_init is not None:
            print('User input parameters may be stuck in local minimum; try a more accurate guess.')
    else:
        stop_MC = True

    return parameter, stop_MC, error

def phase_centered(
        f, fr, Ql, theta, delay=0.):
    return theta - 2 * np.pi * delay * (f - fr) + 2. * np.arctan(2. * Ql * (1. - f / fr))

def phase_dist(
        angle):
    return np.pi - np.abs(np.pi - np.abs(angle))

def fit_phase(
        f_data, 
        z_data, 
        guesses=None):

    phase = np.unwrap(np.angle(z_data))

    if np.max(phase) - np.min(phase) <= 0.8 * 2 * np.pi:
        logging.warning(
            "Data does not cover a full circle."
            "Increase the frequency span?"
        )
        roll_off = np.max(phase) - np.min(phase)
    else:
        roll_off = 2 * np.pi

    if guesses is None:
        phase_smooth      = gaussian_filter1d(phase, 30)
        phase_derivative  = np.gradient(phase_smooth)
        fr_guess          = f_data[np.argmax(np.abs(phase_derivative))]
        Ql_guess          = 2 * fr_guess / (f_data[-1] - f_data[0])
        slope             = phase[-1] - phase[0] + roll_off
        delay_guess       = -slope / (2 * np.pi * (f_data[-1] - f_data[0]))
    else:
        fr_guess, Ql_guess, delay_guess = guesses

    theta_guess = (np.mean(phase[:3]) + np.mean(phase[-3:])) / 2

    def residuals_full(params):
        return phase_dist(phase - phase_centered(f_data, *params))

    def residuals_Ql(params):
        Ql, = params
        return residuals_full((fr_guess, Ql, theta_guess, delay_guess))

    def residuals_fr_theta(params):
        fr, theta = params
        return residuals_full((fr, Ql_guess, theta, delay_guess))

    def residuals_delay(params):
        delay, = params
        return residuals_full((fr_guess, Ql_guess, theta_guess, delay))

    def residuals_fr_Ql(params):
        fr, Ql = params
        return residuals_full((fr, Ql, theta_guess, delay_guess))

    p_final = spopt.leastsq(residuals_Ql,        [Ql_guess])
    Ql_guess, = p_final[0]
    p_final = spopt.leastsq(residuals_fr_theta,  [fr_guess, theta_guess])
    fr_guess, theta_guess = p_final[0]
    p_final = spopt.leastsq(residuals_delay,     [delay_guess])
    delay_guess, = p_final[0]
    p_final = spopt.leastsq(residuals_fr_Ql,     [fr_guess, Ql_guess])
    fr_guess, Ql_guess = p_final[0]
    p_final = spopt.leastsq(residuals_full, [fr_guess, Ql_guess, theta_guess, delay_guess])

    return p_final[0]

def fit_delay(
        xdata: np.ndarray, 
        ydata: np.ndarray):
    
    xc, yc, r0 = find_circle(np.real(ydata), np.imag(ydata))
    z_data      = ydata - complex(xc, yc)
    fr, Ql, theta, delay = fit_phase(xdata, z_data)

    delay       = delay
    delay_corr  = 0
    residuals   = 0

    for _ in range(10):
        z_data       = ydata * np.exp(2j * np.pi * delay * xdata)
        xc, yc, r0   = find_circle(np.real(z_data), np.imag(z_data))
        z_data       = z_data - complex(xc, yc)

        guesses = (fr, Ql, delay)
        fr, Ql, theta, delay_corr = fit_phase(xdata, z_data, guesses)

        phase_fit = phase_centered(xdata, fr, Ql, theta, delay_corr)
        residuals = np.unwrap(np.angle(z_data)) - phase_fit

        if 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr <= np.std(residuals):
            break

        if delay_corr * delay < 0:
            if abs(delay_corr) > abs(delay):
                delay = delay * 0.5
            else:
                delay = delay * 0.1 * np.sign(delay_corr) * 5e-11
        else:
            if abs(delay_corr) >= 1e-8:
                delay = delay + min(delay_corr, delay)
            elif abs(delay_corr) >= 1e-9:
                delay = delay * 1.1
            else:
                delay = delay + delay_corr

    if 2 * np.pi * (xdata[-1] - xdata[0]) * delay_corr > np.std(residuals):
        logging.warning("Delay could not be fit properly!")

    return delay

def periodic_boundary(
        angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calibrate(
        x_data: np.ndarray, z_data: np.ndarray):
    xc, yc, r = find_circle(np.real(z_data), np.imag(z_data))
    zc = complex(xc, yc)
    z_data2 = z_data - zc

    fr, Ql, theta, delay_remaining = fit_phase(x_data, z_data2)
    beta       = periodic_boundary(theta - np.pi)
    offrespoint = zc + r * np.cos(beta) + 1j * r * np.sin(beta)
    a      = np.absolute(offrespoint)
    alpha  = np.angle(offrespoint)
    phi    = periodic_boundary(beta - alpha)

    r /= a
    return delay_remaining, a, alpha, theta, phi, fr, Ql

def normalize(
        f_data, z_data, delay, a, alpha):
    return (z_data / a) * np.exp(1j * (-alpha))

def remove_f_dep_background(
        xdata, 
        ydata, 
        plot_result=True):
    """
    Remove approximately linear frequency-dependent complex background using a few points from both wings.
    """
    num_points = 3
    x_wing = np.concatenate((xdata[:num_points], xdata[-num_points:]))
    y_wing = np.concatenate((ydata[:num_points], ydata[-num_points:]))

    offset_mag = np.mean(np.abs(y_wing))
    offset_angle = np.mean(np.angle(y_wing))

    p_real      = np.polyfit(x_wing, np.real(y_wing), 1)
    p_imag      = np.polyfit(x_wing, np.imag(y_wing), 1)
    background  = np.polyval(p_real, xdata) + 1j * np.polyval(p_imag, xdata)

    ydata_remove_bg = ydata / background
    mag   = np.abs(ydata_remove_bg) * offset_mag
    angle = np.angle(ydata_remove_bg) + offset_angle
    # angle = np.angle(ydata)
    # angle = np.angle(ydata_remove_bg)
    ydata_remove_bg = mag * np.exp(1j * angle)
    # ydata_remove_bg = ydata

    if plot_result:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(xdata, 20 * np.log10(np.abs(ydata)),           label='Original')
        plt.plot(xdata, 20 * np.log10(np.abs(ydata_remove_bg)), label='Background removed')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.title('Log-Magnitude')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(xdata, np.angle(ydata),           label='Original')
        plt.plot(xdata, np.angle(ydata_remove_bg), label='Corrected')
        plt.xlabel('Frequency')
        plt.ylabel('Phase (rad)')
        plt.title('Phase')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return xdata, ydata_remove_bg

def preprocess_circle(
        xdata: np.ndarray, 
        ydata: np.ndarray, 
        output_path: str, 
        plot_extra):
    """Circle-based preprocessing with frequency-dependent background removal."""
    xdata, ydata = remove_f_dep_background(xdata, ydata, plot_result=plot_extra)

    delay  = fit_delay(xdata, ydata)
    z_data = ydata * np.exp(2j * np.pi * delay * xdata)

    delay_remaining, a, alpha, theta, phi, fr, Ql = calibrate(xdata, z_data)
    z_norm = normalize(xdata, z_data, delay_remaining, a, alpha)

    return z_norm

def background_removal(
        databg, 
        linear_amps: np.ndarray,
        phases: np.ndarray, 
        output_path: str):
    x_bg           = databg.freqs
    linear_amps_bg = databg.linear_amps
    phases_bg      = databg.phases

    fmag = interp1d(x_bg, linear_amps_bg, kind='cubic')
    fang = interp1d(x_bg, phases_bg,      kind='cubic')

    fp.plot2(databg.freqs, databg.linear_amps, x_bg, linear_amps_bg, "VS_mag", output_path)
    fp.plot2(databg.freqs, databg.phases,      x_bg, phases_bg,      "VS_ang", output_path)

    linear_amps = np.divide(linear_amps, linear_amps_bg)
    phases      = np.subtract(phases, phases_bg)

    return np.multiply(linear_amps, np.exp(1j * phases))

def min_fit(
        params, xdata, ydata, Method):
    """
    Minimise DCM fit parameters via lmfit least-squares.

    Parameters
    ----------
    params : lmfit.Parameters
        Initial parameter guess (Q, Qc, w1, phi).
    xdata : np.ndarray
        Frequency data.
    ydata : np.ndarray (complex)
        S21 data.
    Method : FitMethod
        Must be DCM.

    Returns
    -------
    fit_params : list or None
    conf_array : list of 6 floats  [Q, Qi, Qc, Qc_Re, phi, w1]
    """
    if Method.method != 'DCM':
        raise ValueError(f"min_fit: unsupported method '{Method.method}'. Only 'DCM' is supported.")

    try:
        minner = lmfit.Minimizer(ff.min_one_Cavity_dip, params, fcn_args=(xdata, ydata))
        result = minner.minimize(method='least_squares')

        parameter  = result.params.valuesdict()
        fit_params = [value for _, value in parameter.items()]
    except Exception as e:
        print(f">Failed to minimise data for least-squares fit: {e}")
        print(">Confidence intervals unknown and set to 0.0")
        return None, [0, 0, 0, 0, 0, 0]

    # Confidence intervals
    try:
        p_names = [name for name in params if name not in Method.MC_fix]
        ci = lmfit.conf_interval(minner, result, p_names=p_names, sigmas=[2])

        # Q confidence
        Q_conf = max(
            np.abs(ci['Q'][1][1] - ci['Q'][0][1]),
            np.abs(ci['Q'][1][1] - ci['Q'][2][1])
        )

        # |Qc| confidence
        Qc_conf = max(
            np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]),
            np.abs(ci['Qc'][1][1] - ci['Qc'][2][1])
        )
        if np.isinf(Qc_conf):
            Qc_conf = min(
                np.abs(ci['Qc'][1][1] - ci['Qc'][0][1]),
                np.abs(ci['Qc'][1][1] - ci['Qc'][2][1])
            )

        # 1/Re[1/Qc] confidence
        Qc_Re     = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][1][1])
        Qc_Re_neg = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][0][1])
        Qc_Re_pos = 1 / np.real(np.exp(1j * fit_params[3]) / ci['Qc'][2][1])
        Qc_Re_conf = max(np.abs(Qc_Re - Qc_Re_neg), np.abs(Qc_Re - Qc_Re_pos))
        if np.isinf(Qc_Re_conf):
            Qc_Re_conf = min(np.abs(Qc_Re - Qc_Re_neg), np.abs(Qc_Re - Qc_Re_pos))

        # phi confidence
        phi_conf = max(
            np.abs(ci['phi'][1][1] - ci['phi'][0][1]),
            np.abs(ci['phi'][1][1] - ci['phi'][2][1])
        )
        if np.isinf(phi_conf):
            phi_conf = min(
                np.abs(ci['phi'][1][1] - ci['phi'][0][1]),
                np.abs(ci['phi'][1][1] - ci['phi'][2][1])
            )

        # w1 (resonance frequency) confidence
        w1_conf = max(
            np.abs(ci['w1'][1][1] - ci['w1'][0][1]),
            np.abs(ci['w1'][1][1] - ci['w1'][2][1])
        )
        if np.isinf(w1_conf):
            # BUG FIX: result was previously computed but never assigned
            w1_conf = min(
                np.abs(ci['w1'][1][1] - ci['w1'][0][1]),
                np.abs(ci['w1'][1][1] - ci['w1'][2][1])
            )

        # Qi uncertainty via error propagation
        Q_ufloat   = ufloat(ci['Q'][1][1],   Q_conf)
        Qc_ufloat  = ufloat(ci['Qc'][1][1],  Qc_conf)
        phi_ufloat = ufloat(ci['phi'][1][1], phi_conf)
        Qi_ufloat  = (1 / Q_ufloat - umath.cos(phi_ufloat) / Qc_ufloat) ** -1

        conf_array = [Q_conf, Qi_ufloat.s, Qc_conf, Qc_Re_conf, phi_conf, w1_conf]

    except Exception as e:
        print(f">{e}")
        print(">Failed to find confidence intervals for least-squares fit")
        conf_array = [0, 0, 0, 0, 0, 0]

    return fit_params, conf_array

def fit(
        resonator):
    """
    Fit a DCM resonator.

    Parameters
    ----------
    resonator : Resonator
        Populated Resonator object with data, method_class, etc.

    Returns
    -------
    output_params : list   [Q, Qc, w1, phi]
    conf_array    : list   [Q, Qi, Qc, Qc_Re, phi, w1] confidence intervals
    error         : float  Monte Carlo fit error
    init          : list   initial guess parameters
    output_path   : str    folder where plots were saved
    """
    filepath          = resonator.filepath
    Method            = resonator.method_class
    normalize_pts     = resonator.normalize
    data              = resonator.data
    plot_extra        = resonator.plot_extra
    preprocess_method = resonator.preprocess_method

    # Resolve output directory
    if filepath is not None:
        dir, filename = os.path.split(filepath)
        if dir == '':
            dir = ROOT_DIR
    else:
        dir      = ROOT_DIR
        filename = 'scresonators'

    # Load data
    try:
        xdata        = data.freqs
        linear_amps  = data.linear_amps
        phases       = np.unwrap(data.phases)
        ydata        = np.multiply(linear_amps, np.exp(1j * phases))
    except Exception as e:
        raise ValueError(f"Failed to read resonator data: {e}")

    output_path = fp.name_folder(dir, str(Method.method))
    if plot_extra:
        # BUG FIX: exist_ok=True prevents crash if folder already exists
        os.makedirs(output_path, exist_ok=True)

    x_initial = xdata
    y_initial = ydata

    # Preprocessing / background normalisation
    slope = intercept = slope2 = intercept2 = 0
    if resonator.databg is not None:
        ydata = background_removal(resonator.databg, linear_amps, phases, output_path)
    elif preprocess_method == "linear":
        ydata, slope, intercept, slope2, intercept2 = preprocess_linear(
            xdata, ydata, normalize_pts, output_path, plot_extra
        )
    elif preprocess_method == "circle":
        ydata = preprocess_circle(xdata, ydata, output_path, plot_extra)

    y_raw = ydata
    x_raw = xdata

    # ── DCM-only parameter vary flags ─────────────────────────────────────
    manual_init = Method.manual_init
    change_Q    = 'Q'   not in Method.MC_fix
    change_Qc   = 'Qc'  not in Method.MC_fix
    change_w1   = 'w1'  not in Method.MC_fix
    change_phi  = 'phi' not in Method.MC_fix

    y1data = np.real(ydata)
    y2data = np.imag(ydata)

    # ── Step 1: Initial guess ──────────────────────────────────────────────
    init = [0] * 4

    if manual_init is not None:
        try:
            if len(manual_init) != 4:
                raise ValueError(
                    "manual_init must have exactly 4 elements: [Q, Qc, f_c, phi]"
                )
            # BUG FIX: kappa moved inside the if block so it is set after
            # manual_init[0] is updated, avoiding the previous divide-by-zero.
            Qc_complex      = manual_init[1] / np.exp(1j * manual_init[3])
            manual_init[0]  = 1 / (1 / manual_init[0] + np.real(1 / Qc_complex))
            kappa           = manual_init[2] / manual_init[0]

            init   = manual_init
            freq   = init[2]
            x_c = y_c = r = 0
            print(f"Manual initial guess: {manual_init}")
        except Exception as e:
            print(f"Exception loading manual_init: {e}")
            raise ValueError(
                "Problem loading manually initialised parameters. "
                "Ensure all values are numeric and [Q, Qc, f_c, phi] format."
            )
    else:
        init, x_c, y_c, r = find_initial_guess(
            xdata, y1data, y2data, Method, output_path, plot_extra
        )
        freq  = init[2]
        kappa = init[2] / init[0]

    # ── Step 2: Least-squares minimisation ────────────────────────────────
    xdata, ydata = x_raw, y_raw

    try:
        params = lmfit.Parameters()
        params.add('Q',   value=init[0], vary=change_Q,   min=init[0] * 0.5, max=init[0] * 1.5)
        params.add('Qc',  value=init[1], vary=change_Qc,  min=init[1] * 0.3, max=init[1] * 1.3)
        params.add('w1',  value=init[2], vary=change_w1,  min=init[2] * 0.5, max=init[2] * 1.5)
        params.add('phi', value=init[3], vary=change_phi, min=-np.pi,          max=np.pi)
    except Exception as e:
        raise ValueError(f"Failed to define lmfit parameters: {e}")

    fit_params, conf_array = min_fit(params, xdata, ydata, Method)

    if fit_params is None:
        if manual_init is None:
            raise RuntimeError("Failed to minimise function for least-squares fit.")
        fit_params = manual_init

    # ── Step 3: Monte Carlo refinement ────────────────────────────────────
    MC_counts        = 0
    error            = [10]
    stop_MC          = False
    output_params    = []
    continue_condition = (MC_counts < Method.MC_iteration) and not stop_MC

    while continue_condition:
        MC_param, stop_MC, error_MC = monte_carlo_fit(xdata, ydata, fit_params, Method)
        error.append(error_MC)
        if error[MC_counts] < error_MC:
            stop_MC = True

        output_params.append(MC_param)
        MC_counts += 1
        continue_condition = (MC_counts < Method.MC_iteration) and not stop_MC

        if not continue_condition:
            output_params = output_params[MC_counts - 1]

    error = min(error)

    # If MC improved the fit, run a final minimisation on MC parameters
    if output_params[0] != fit_params[0]:
        try:
            params2 = lmfit.Parameters()
            params2.add('Q',   value=output_params[0], vary=change_Q,
                        min=output_params[0] * 0.5, max=output_params[0] * 1.5)
            params2.add('Qc',  value=output_params[1], vary=change_Qc,
                        min=output_params[1] * 0.8, max=output_params[1] * 1.2)
            params2.add('w1',  value=output_params[2], vary=change_w1,
                        min=output_params[2] * 0.9, max=output_params[2] * 1.1)
            params2.add('phi', value=output_params[3], vary=change_phi,
                        min=output_params[3] * 0.9, max=output_params[3] * 1.1)
            output_params, conf_array = min_fit(params2, xdata, ydata, Method)
        except Exception as e:
            print(f"Warning: post-MC re-minimisation failed: {e}")

    if len(xdata) == 0:
        if manual_init is not None:
            print(">Extracted data length is zero — initial parameters may be too far off.")
        else:
            raise ValueError(
                ">Extracted data length is zero. Please manually supply an initial guess."
            )

    # ── Step 4: Plot and save ──────────────────────────────────────────────
    if resonator.plot is not None:
        os.makedirs(output_path, exist_ok=True)

        kappa          = output_params[2] / output_params[0]
        xstart         = output_params[2] - kappa / 2
        xend           = output_params[2] + kappa / 2
        extract_factor = [xstart, xend]

        try:
            figurename = f"DCM with Monte Carlo Fit and Raw data\nPower: {filename}"
            title      = "DCM Method Fit"
            fig = fp.PlotFit(
                x_raw, y_raw, x_initial, y_initial,
                slope, intercept, slope2, intercept2,
                output_params, Method, error, figurename,
                x_c, y_c, r, output_path, conf_array,
                extract_factor, title=title,
                manual_params=Method.manual_init
            )
        except Exception as e:
            raise RuntimeError(f"Failed to plot DCM fit: {e}")

        try:
            fig.savefig(
                fp.name_plot(filename, str(Method.method), output_path,
                             format=f'.{resonator.plot}'),
                format=f'{resonator.plot}'
            )
        except Exception as e:
            raise ValueError(
                f"Unrecognised file format '{resonator.plot}'. "
                f"Please use png, pdf, ps, eps or svg. ({e})"
            )

    return output_params, conf_array, error, init, output_path