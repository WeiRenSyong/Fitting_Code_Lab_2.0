import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import uncertainties
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

import fit_resonator.cavity_functions as ff

# NOTE: fit.py is NOT imported here to break the circular dependency.
# The fit function reference is passed in as Method.func where needed.


def name_plot(
        filename, 
        strmethod, 
        output_path, 
        format='.pdf'):

    if filename.endswith('.csv'):
        filename = filename[:-4]
    filename = filename.replace('.', 'p')
    
    return output_path + strmethod + '_' + filename + format

def plot(
        x, 
        y, 
        name, 
        output_path, 
        x_c=None, 
        y_c=None, 
        r=None, 
        p_x=None, 
        p_y=None):
    """Plot any given set of x, y data with optional circle and resonance marker."""
    fig = plt.figure(figsize=(10, 10))   # BUG FIX: no shared figure name
    gs  = GridSpec(2, 2)
    ax  = plt.subplot(gs[0:2, 0:2])

    ax.plot(x, y, 'bo', label='raw data', markersize=3)

    if x_c is not None and y_c is not None and r is not None:
        circle = Circle((x_c, y_c), r, facecolor='none',
                        edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)

    if p_x is not None and p_y is not None:
        ax.plot(p_x, p_y, '*', color='red', markersize=5)

    plt.gca().set_aspect('equal')
    fig.savefig(output_path + name + '.pdf', format='pdf')
    plt.close(fig)

def plot2(
        x, 
        y, 
        x2, 
        y2, 
        name, 
        output_path):
    """Plot two datasets for comparison."""
    fig = plt.figure(figsize=(10, 10))   # BUG FIX: no shared figure name
    gs  = GridSpec(2, 2)
    ax  = plt.subplot(gs[0:2, 0:2])
    ax.plot(x,  y,  'bo', label='data 1', markersize=3)
    ax.plot(x2, y2, 'ro', label='data 2', markersize=3)
    ax.legend()
    fig.savefig(output_path + name + '.pdf', format='pdf')
    plt.close(fig)

def name_folder(
        dir, 
        strmethod):
    """Generate a timestamped output folder path."""
    result = time.localtime(time.time())
    output = strmethod + '_' + str(result.tm_year)

    # Zero-pad month
    if len(str(result.tm_mon)) < 2:
        output += '0' + str(result.tm_mon)
    else:
        output += str(result.tm_mon)

    # BUG FIX: was `if len(str(result.tm_mday)):` which is always True
    if len(str(result.tm_mday)) < 2:
        output += '0' + str(result.tm_mday)
    else:
        output += str(result.tm_mday)

    output += '_' + str(result.tm_hour) + '_' + str(result.tm_min) + '_' + str(result.tm_sec)

    output_path = (dir + '/' if dir is not None else '') + output + '/'

    # Avoid collision with existing folders
    count = 2
    base  = output_path
    while os.path.isdir(output_path):
        output_path = base[:-1] + '_' + str(count) + '/'
        count += 1

    return output_path

def create_metadata(
        Method, 
        output_path):
    """Write fit configuration to a metadata CSV."""
    with open(output_path + "metadata.csv", "w", newline='') as file:
        writer = csv.writer(file)
        fields = ['Method', 'MC_iteration', 'MC_rounds',
                  'MC_weight', 'MC_weightvalue', 'MC_fix',
                  'MC_step_const', 'manual_init', 'preprocess_method']
        vals   = [Method.method, Method.MC_iteration, Method.MC_rounds,
                  Method.MC_weight, Method.MC_weightvalue, Method.MC_fix,
                  Method.MC_step_const, Method.manual_init, Method.preprocess_method]
        writer.writerow(fields)
        writer.writerow(vals)
    # BUG FIX: removed redundant file.close() — 'with' handles it

def round_sigfig(v, n):
    if not np.isfinite(v) or v == 0:
        return 0.0
    return round(v, n - int(np.floor(np.log10(abs(v)))) - 1)

def PlotFit(
        x, 
        y, 
        x_initial, 
        y_initial,
        slope, 
        intercept, 
        slope2, 
        intercept2,
        params, 
        Method, 
        error, 
        figurename,
        x_c, 
        y_c, 
        radius, 
        output_path,
        conf_array, 
        extract_factor=None, 
        title="DCM Fit",
        manual_params=None,
        dfac: int = 1,
        msizes: list = None,
        xstr: str = r'$(f-f_c)$ [kHz]',
        fscale: float = 1e3,
        fsize: float = 20.):
    """
    Plot DCM fit results and write fit_params.csv to output_path.

    Parameters
    ----------
    x, y               : frequency and (complex) S21 data after normalisation
    x_initial, y_initial: original data before normalisation
    slope/intercept/slope2/intercept2 : linear normalisation coefficients
    params             : [Q, Qc, w1, phi] fit results
    Method             : FitMethod instance (DCM)
    error              : Monte Carlo fit error
    figurename         : figure window name
    x_c, y_c, radius   : circle centre and radius from initial guess
    output_path        : directory to save outputs
    conf_array         : [Q, Qi, Qc, Qc_Re, phi, w1] confidence intervals
    extract_factor     : [xstart, xend] frequency window, or None
    manual_params      : user-supplied initial guess, or None
    dfac               : decimation factor for circle plot points
    msizes             : [marker_size_data, marker_size_resonance]
    xstr               : x-axis label string
    fscale             : frequency axis scale factor (default 1e3 → kHz)
    fsize              : font size
    """
    if msizes is None:
        msizes = [12, 24]

    plt.close(figurename)

    func  = Method.func
    fcscale = 1. if params[2] < 1e9 else 1e9

    # Generate dense frequency array for fit curve
    if extract_factor is not None and isinstance(extract_factor, list):
        x_fit = np.linspace(extract_factor[0], extract_factor[1], 5000)
    else:
        x_fit = np.linspace(x.min(), x.max(), 5000)
    y_fit = func(x_fit, *params)
    # Always use full range for plotting
    x_fit = np.linspace(x.min(), x.max(), 5000)
    y_fit = func(x_fit, *params)

    fig = plt.figure(figurename, figsize=(18, 12))
    gs  = GridSpec(11, 10)
    ax0 = plt.subplot(gs[1:10, 0:6])
    ax1 = plt.subplot(gs[0:4,  7:10])
    ax2 = plt.subplot(gs[4:8,  7:10])
    fig.set_tight_layout(True)

    msize1, msize2 = msizes

    # ── Manual parameter display (DCM only) ───────────────────────────────
    if manual_params is not None:
        Qc_man = manual_params[1] / np.exp(1j * manual_params[3])
        Qi_man = (manual_params[0] ** -1 - abs(np.real(Qc_man ** -1))) ** -1
        textstr = (
            r'Manually input parameters:' + '\n'
            + 'Q = '                    + '%s' % float('{0:.5g}'.format(manual_params[0]))
            + '\n' + r'1/Re[1/$Q_c$] = ' + '%s' % float('{0:.5g}'.format(1 / np.real(1 / Qc_man)))
            + '\n' + r'$Q_c$ = '          + '%s' % float('{0:.5g}'.format(manual_params[1]))
            + '\n' + r'$Q_i$ = '          + '%s' % float('{0:.5g}'.format(Qi_man))
            + '\n' + r'$f_c$ = '          + '%s' % float('{0:.5g}'.format(manual_params[2] / fcscale)) + ' GHz'
            + '\n' + r'$\phi$ = '         + '%s' % float('{0:.5g}'.format(manual_params[3])) + ' radians'
        )
        plt.gcf().text(0.1, 0.7, textstr, fontsize=15)

    # Round x_initial to avoid floating-point noise
    x_initial = np.round(x_initial, 8)

    # Decimate data for circle plot
    x = x[0::dfac]
    y = y[0::dfac]

    # ── Axis labels (DCM only) ────────────────────────────────────────────
    ax1.set_ylabel('Mag[S21]',  fontsize=fsize)
    ax2.set_ylabel('Ang[S21]',  fontsize=fsize)
    ax0.set_ylabel(r'Im[$S_{21}$]', fontsize=fsize)
    ax0.set_xlabel(r'Re[$S_{21}$]', fontsize=fsize)

    # ── Resonance circle plot ─────────────────────────────────────────────
    ax0.plot(np.real(y),     np.imag(y),     'bo',
             label='normalized data', markersize=msize1)
    ax0.plot(np.real(y_fit), np.imag(y_fit), color='cadetblue',
             label='fit function', linewidth=4)
    ax0.axhline(y=0., color='k')
    ax0.axvline(x=1., color='k')
    ax0.set_aspect(1.)
    leg = ax0.legend(loc="upper left", fancybox=True, shadow=True, fontsize=20)

    # ── Magnitude and phase subplots ──────────────────────────────────────
    ax1.plot((x     - params[2]) / fscale, np.log10(np.abs(y))     * 20,
             'bo', label='normalized data', markersize=msize1)
    ax1.plot((x_fit - params[2]) / fscale, np.log10(np.abs(y_fit)) * 20,
             color='cadetblue', lw=4, label='fit function')
    ax1.set_xlim((x[0]  - params[2]) / fscale, (x[-1] - params[2]) / fscale)
    ax1.set_xlabel(xstr, fontsize=fsize)

    ax2.plot((x     - params[2]) / fscale, np.angle(y),
             'bo', label='normalized data', markersize=msize1)
    ax2.plot((x_fit - params[2]) / fscale, np.angle(y_fit),
             color='cadetblue', label='fit function', lw=4)
    ax2.set_xlim((x[0]  - params[2]) / fscale, (x[-1] - params[2]) / fscale)
    ax2.set_xlabel(xstr, fontsize=fsize)

    # ── DCM resonance point: S21(f_c) = 1 - Q/Qc * exp(i*phi) ───────────
    resonance = 1 - params[0] / params[1] * np.exp(1j * params[3])
    ax0.plot(np.real(resonance), np.imag(resonance), '*',
             color='darkorange', label='resonance', markersize=msize2)
    ax1.plot(0, np.log10(np.abs(resonance)) * 20, '*',
             color='darkorange', label='resonance', markersize=msize2)
    ax2.plot(0, np.angle(resonance), '*',
             color='darkorange', label='resonance', markersize=msize2)

    for line in leg.get_lines():
        line.set_linewidth(10)

    # ── Parameter text box and CSV output ────────────────────────────────
    try:
        if params:
            Qc_complex = params[1] / np.exp(1j * params[3])
            Qi         = (params[0] ** -1 - np.real(Qc_complex ** -1)) ** -1
            Qc_Re      = 1 / np.real(1 / Qc_complex)

            if Qi < 0:
                print(
                    "Warning: Qi < 0. Check data format (dB magnitude, radians phase) "
                    "and that this is a notch-type resonator."
                )
            if Qc_Re < 0:
                print("Warning: 1/Re[1/Qc] < 0. Calculating Qi anyway.")

            Q    = params[0]
            Qc   = params[1]
            f_c  = params[2]
            phi  = params[3]

            reports = ['Q', r'$Q_i$', r'$|Q_c|$', r'_', r'$\phi$', r'$f_c$']
            p_ref   = [Q, Qi, Qc, None, phi, f_c]

            textstr = ''
            for val in reports:
                if val == r'_':
                    continue    # placeholder to keep conf_array indices aligned
                vscale = fcscale if val == r'$f_c$' else 1.
                idx    = reports.index(val)
                err = round_sigfig(conf_array[idx] / vscale, 1)
                v   = p_ref[idx] / vscale

                if not np.isfinite(v):
                    v = 0.0
                if not np.isfinite(err):
                    err = 0.0

                un = uncertainties.ufloat(v, err)
                textstr += r'%s: $%s$' % (val, f'{un:L}')
                if   val == r'$\phi$': textstr += ' radians'
                elif val == r'$f_c$':  textstr += ' GHz'
                textstr += '\n'

            plt.gcf().text(0.63, 0.05, textstr, fontsize=20)

            # Write fit_params.csv
            with open(output_path + "fit_params.csv", "w", newline='') as file:
                writer = csv.writer(file)
                fields = ['Q', 'Qi', '|Qc|', '1/Re[1/Qc]', 'phi', 'fc']
                vals   = [
                    [float('{0:.10g}'.format(Q)),     float('{0:.10g}'.format(Qi)),
                     float('{0:.10g}'.format(Qc)),    float('{0:.10g}'.format(Qc_Re)),
                     float('{0:.10g}'.format(phi)),   float('{0:.10g}'.format(f_c))],
                    [float('{0:.5g}'.format(conf_array[0])), float('{0:.5g}'.format(conf_array[1])),
                     float('{0:.5g}'.format(conf_array[2])), float('{0:.5g}'.format(conf_array[3])),
                     float('{0:.5g}'.format(conf_array[4])), float('{0:.5g}'.format(conf_array[5]))]
                ]
                writer.writerow(fields)
                writer.writerows(vals)
            # BUG FIX: removed redundant file.close()

    except Exception as e:
        print(f">{e}")
        raise RuntimeError(">Error writing parameters on plot")

    # Write metadata
    try:
        create_metadata(Method, output_path)
    except Exception as e:
        print(f">{e}")
        raise RuntimeError(">Error creating metadata file")

    return fig