'''fit'''

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

def find_nearest(
        array, # Input data
        value): # The nearest value and its corresponding index we want to find
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

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
        x, # frequency
        y1, # real part
        y2, # imaginary part
        Method, # DCM
        ):
    """
    Determine initial guess for DCM parameters
    INPUT:  [frequency, real part, imag part, DCM]
    OUTPUT: [Q, Qc, f_c, phi], [x_c, y_c, r]
    """
    if Method.method != "DCM":
        raise ValueError("It currently supports DCM only.")

    try:
        ###########################################################################
        #### Find the complex circle and determine the center and radius of it ####
        ###########################################################################
        y = y1 + 1j * y2
        # When the data is high SNR, smooth_sigma can set to 1
        # When the data is moderate SNR, smooth_sigma can set to 2
        # When the data is low SNR, smooth_sigma can set to 3
        smooth_sigma = 3 
        y1_smooth = gaussian_filter1d(y1, smooth_sigma)
        y2_smooth = gaussian_filter1d(y2, smooth_sigma)
        y_smooth = y1_smooth + 1j * y2_smooth
        x_c, y_c, r = find_circle(y1_smooth, y2_smooth)
        z_c = x_c + 1j * y_c

        
        mag = np.abs(y_smooth)

        # Estimate off-resonance point from both frequency wings
        n_edge = 3
        z_off = np.mean(np.r_[y_smooth[:n_edge], y_smooth[-n_edge:]])

        # Resonance point is approximately opposite the off-resonance point on the circle
        theta_off = np.angle(z_off - z_c)
        z_res_target = z_c + r * np.exp(1j * (theta_off + np.pi))

        # Pick measured point closest to that opposite-circle target
        freq_idx = np.argmin(np.abs(y_smooth - z_res_target))
        f_c_guess = x[freq_idx]   

    except Exception as e:
        raise ValueError(f"Problem in find_circle(): {e}")
    
    try:
        phi = np.angle(-z_c)
        f_c = f_c_guess

        mag_wing = np.r_[mag[:3], mag[-3:]]
        off_mag = np.mean(mag_wing)

        dip_mag = mag[freq_idx]
        depth = off_mag - dip_mag

        if depth <= 0:
            raise RuntimeError("No visible resonance dip in magnitude.")

        Q_over_Qc = depth / off_mag   # Q_over_Qc indicates the how deep of the resonance dip

        half_level = dip_mag + depth / 2

        left_idx = np.argmin(np.abs(mag[:freq_idx] - half_level))
        right_idx = np.argmin(np.abs(mag[freq_idx:] - half_level))


        idx1 = left_idx
        idx2 = freq_idx + right_idx

        if idx2 <= idx1:
            raise RuntimeError("Invalid FWHM index ordering.")

        kappa = abs(x[idx2] - x[idx1])   # linewidth of the resonance

        if kappa <= 0:
            raise RuntimeError("Invalid linewidth from FWHM.")
        
        Q = f_c / kappa
        Qc = Q / Q_over_Qc  

        print(f'Initial guess before curve fit shows that fc = {f_c/1e9:.6f} GHz')
        print(f'Initial guess before curve fit shows that linewidth = {kappa/1e3:.3f} kHz.')
        print(f'Initial guess before curve fit shows that Q = {Q:.0f}.')
        print(f'Initial guess before curve fit shows that Qc = {Qc:.0f}.')   

        fit_mask = (x > f_c - 5 * kappa) & (x < f_c + 5 * kappa)
        if np.sum(fit_mask) < 5:
            raise RuntimeError("Not enough points for magnitude curve_fit refinement.")
        
        popt, _ = spopt.curve_fit(
            ff.one_cavity_peak_abs,
            x[fit_mask],
            np.abs(y_smooth[fit_mask])/off_mag,
            p0=[Q, Qc, f_c],
            # Give a perturbation to refine initial guess
            bounds=([Q*0.95, Qc*0.95, x[fit_mask].min()], [Q*1.05, Qc*1.05, x[fit_mask].max()])
            # bounds=([1e-10, 1e-10, x[fit_mask].min()], [1e10, 1e10, x[fit_mask].max()])
        )
        Q, Qc, f_c = popt
        kappa = f_c / Q
        init_guess = [Q, Qc, f_c, phi]

        print(f'Initial guess after curve fit shows that fc = {f_c/1e9:.6f} GHz')
        print(f'Initial guess after curve fit shows that linewidth = {kappa/1e3:.3f} kHz.')
        print(f'Initial guess after curve fit shows that Q = {Q:.0f}.')
        print(f'Initial guess after curve fit shows that Qc = {Qc:.0f}.')      

    except Exception as e:
        raise RuntimeError(f"Initial guess failed inside find_initial_guess(): {e}") from e

    return init_guess, x_c, y_c, r

def monte_carlo_fit(
        xdata=None, 
        ydata=None, 
        parameter=None, 
        Method=None):
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
        f, 
        fr, 
        Ql, 
        theta, 
        delay=0.):
    return theta - 2 * np.pi * delay * (f - fr) + 2. * np.arctan(2. * Ql * (1. - f / fr))




def phase_dist(
        angle):
    return np.pi - np.abs(np.pi - np.abs(angle))

def fit_phase(
        f_data, 
        z_data, 
        guesses=None):

    phase = np.unwrap(np.angle(z_data))

    phase_span = np.max(phase) - np.min(phase)

    if phase_span < 0.3:   # <<<< critical threshold (~17 deg)
        logging.warning("Phase span too small — skipping phase-based delay fit")
        return guesses if guesses is not None else (f_data[np.argmax(np.abs(np.gradient(phase)))], 1e4, 0.0, 0.0)

    roll_off = phase_span

    if guesses is None:
        phase_smooth = gaussian_filter1d(phase, 30)
        phase_derivative = np.gradient(phase_smooth)
        fr_guess = f_data[np.argmax(np.abs(phase_derivative))]
        Ql_guess = 2 * fr_guess / (f_data[-1] - f_data[0])
        slope = phase[-1] - phase[0] + roll_off
        delay_guess = -slope / (2 * np.pi * (f_data[-1] - f_data[0]))
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
        ydata):
    n_edge = max(5, len(xdata)//20)
    off_mag = np.mean(np.r_[np.abs(ydata[:n_edge]), np.abs(ydata[-n_edge:])])
    ydata_norm = ydata / off_mag

    return xdata, ydata_norm

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
                delay = delay * 0.9
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

def preprocess_circle(
        xdata: np.ndarray, 
        ydata: np.ndarray, 
        output_path: str, 
        plot_extra):
    """Circle-based preprocessing with frequency-dependent background removal."""
    xdata, ydata = remove_f_dep_background(xdata, ydata)

    # delay  = fit_delay(xdata, ydata)
    phase_span = np.max(np.angle(ydata)) - np.min(np.angle(ydata))

    if phase_span < 0.3:
        delay = 0.0
    else:
        delay = fit_delay(xdata, ydata)

    z_data = ydata * np.exp(2j * np.pi * delay * xdata)

    print(f"[DEBUG] delay = {delay:.3e}")

    # TEMP: disable delay to test
    z_data = ydata  # <-- comment out delay application

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
        params, 
        xdata, 
        ydata, 
        Method):
    """
    Minimize DCM fit parameters via lmfit least-squares.

    Parameters
    ----------
    params : lmfit.Parameters
        Initial parameter guess (Q, Qc, w1, phi).
    xdata : Frequency data (np.ndarray).
    ydata : S21 data (np.ndarray) (complex).
    Method : Must be DCM.

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
        if not result.success:
            print(">Fit did not converge cleanly; confidence intervals set to 0.0")
            return fit_params, [0, 0, 0, 0, 0, 0]

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
        den = np.real(np.exp(1j * fit_params[3]) / ci['Qc'][1][1])
        Qc_Re = 1 / den if den != 0 else np.nan
        den_neg = np.real(np.exp(1j * fit_params[3]) / ci['Qc'][0][1])
        den_pos = np.real(np.exp(1j * fit_params[3]) / ci['Qc'][2][1])
        Qc_Re_neg = 1 / den_neg if den_neg != 0 else np.nan
        Qc_Re_pos = 1 / den_pos if den_pos != 0 else np.nan
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
        phases       = data.phases

        sort_idx = np.argsort(xdata)
        xdata = xdata[sort_idx]
        linear_amps = linear_amps[sort_idx]
        phases = phases[sort_idx]

        ydata = np.multiply(linear_amps, np.exp(1j * phases))
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
            xdata, y1data, y2data, Method,
        )
        freq  = init[2]
        kappa = init[2] / init[0]

    # ── Step 2: Least-squares minimisation ────────────────────────────────
    xdata, ydata = x_raw, y_raw
    
    w1_window = 5 * kappa
    try:
        params = lmfit.Parameters()
        params.add('Q',  value=init[0], vary=change_Q,   min=init[0] * 0.1, max=init[0] * 10.0)

        params.add('Qc', value=init[1], vary=change_Qc, min=init[1] * 0.1, max=init[1] * 10.0)
        params.add(
            'w1',
            value=init[2],
            vary=change_w1,
            min=max(np.min(xdata), init[2] - w1_window),
            max=min(np.max(xdata), init[2] + w1_window)
        )
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
    if np.isclose(output_params[2], np.min(xdata)) or np.isclose(output_params[2], np.max(xdata)):
        print("[WARNING] fitted fc is at sweep boundary → bad fit")

    # If MC improved the fit, run a final minimisation on MC parameters
    if output_params[0] != fit_params[0]:
        try:
            params2 = lmfit.Parameters()
            params2.add('Q',  value=output_params[0], vary=change_Q, min=output_params[0] * 0.1, max=output_params[0] * 10.0)
            params2.add('Qc', value=output_params[1], vary=change_Qc, min=output_params[1] * 0.1, max=output_params[1] * 10.0)
            w1_window = 5 * kappa

            params2.add(
                'w1',
                value=output_params[2],
                vary=change_w1,
                min=max(np.min(xdata), init[2] - w1_window),
                max=min(np.max(xdata), init[2] + w1_window)
            )
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
            safe_params = [
                float(v) if np.isfinite(v) else 0.0
                for v in output_params
            ]

            safe_conf = [
                float(v) if np.isfinite(v) else 0.0
                for v in conf_array
            ]

            fig = fp.PlotFit(
                x_raw, y_raw, x_initial, y_initial,
                slope, intercept, slope2, intercept2,
                safe_params, Method, error, figurename,
                x_c, y_c, r, output_path, safe_conf,
                extract_factor, title=title,
                manual_params=Method.manual_init
            )
            resonator.plot = 'png' if getattr(resonator, "save_dcm_plot", False) else 'png'
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
        
    print("=== Raw data diagnostic ===")
    print(f"x range: {xdata.min()/1e9:.6f} to {xdata.max()/1e9:.6f} GHz")
    print(f"phase min/max before unwrap: {data.phases.min():.3f}, {data.phases.max():.3f}")
    print(f"phase min/max after unwrap: {phases.min():.3f}, {phases.max():.3f}")
    print(f"linear amp min/max: {linear_amps.min():.3e}, {linear_amps.max():.3e}")

    return output_params, conf_array, error, init, output_path