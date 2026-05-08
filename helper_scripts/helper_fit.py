
'''helper_fit.py'''

import os
import sys
import datetime
from pathlib import Path

import regex
import uncertainties

# Make local copied folders take priority
THIS_FILE = Path(__file__).resolve()
HELPER_DIR = THIS_FILE.parent
BASE_DIR = HELPER_DIR.parent

sys.path.insert(0, str(HELPER_DIR))
sys.path.insert(0, str(BASE_DIR / "scresonators"))
sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

import fit_resonator.resonator as res
import fit_resonator.fit as fsd
import helper_misc as hm

print("helper_fit loaded from:", THIS_FILE)
print("helper_misc loaded from:", hm.__file__)
print("fit_resonator.resonator loaded from:", res.__file__)
print("fit_resonator.fit loaded from:", fsd.__file__)

def fit_single_res(
        filename, 
        preprocess_method='circle',
        save_dcm_plot=False,
        save_fit_dirs=r"fits/", 
        manual_init=None, 
        plot_extra=False,):
    """
    Fit a single resonator from file.
    """

    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Resonator file not found: {file_path}")
    print("-------------")
    print(f"fit_single_res: {file_path}")

    # Fit settings
    fit_type = 'DCM'
    MC_iteration = 10
    MC_rounds = 10
    MC_fix = []
    normalize = 5

    myres = res.Resonator()
    myres.from_file(str(file_path))
    myres.preprocess_method = preprocess_method
    myres.normalize = normalize
    myres.save_dcm_plot = save_dcm_plot
    myres.plot_extra = plot_extra
    myres.plot = 'png'

    if isinstance(save_fit_dirs, (list, tuple)):
        myres.fit_dir = save_fit_dirs[0]
    else:
        myres.fit_dir = save_fit_dirs

    myres.fit_method(
        fit_type,
        MC_iteration,
        MC_rounds=MC_rounds,
        MC_fix=MC_fix,
        manual_init=manual_init,
        MC_step_const=0.3
    )

    fig = None

    fit_result = fsd.fit(myres)
    if fit_result is None:
        raise RuntimeError(
            f"fsd.fit returned None for file: {file_path}. "
            "Fitting failed internally (likely bad initial guess or circle preprocessing failed).")
    params, conf_intervals, err, init1, fig = fit_result

    return params, err, conf_intervals, fig

def fit_single_power_list(
        filenames,
        powers,
        preprocess_method='circle',
        phi0=0.,
        show_plots=False,
        save_dcm_plot=True,
        save_fit_dirs=r"fits/",
        manual_init_list=None,
        plot_extra=False):
    """
    Fit each power independently and return fitted parameters.
    """

    if manual_init_list is None:
        manual_init_list = [None] * len(filenames)

    results = []

    for idx, filename in enumerate(filenames):
        print(f"\n[SINGLE POWER FIT] {filename}")

        params, err, conf_int, fig = fit_single_res(
            filename,
            preprocess_method=preprocess_method,
            save_dcm_plot=save_dcm_plot,
            save_fit_dirs=save_fit_dirs,
            manual_init=manual_init_list[idx],
            plot_extra=plot_extra
        )

        Qcj = params[1] * np.exp(1j * (params[3] + phi0))
        Qij = 1.0 / (1.0 / params[0] - np.real(1.0 / Qcj))

        results.append({
            "filename": filename,
            "power": powers[idx],
            "params": params,
            "err": err,
            "conf_int": conf_int,
            "Q": params[0],
            "Qi": np.real(Qij),
            "Qc": np.real(Qcj),
            "fc": params[2],
            "phi": params[3],
            "fig": fig,
        })

        if show_plots:
            fig.show()
        else:
            plt.close('all')

    return results

def fit_qiqcfc_vs_power(
        filenames=None, 
        powers=None,
        single_fit_results=None,
        atten=[0, 70],
        preprocess_method='circle', 
        phi0=0., 
        data_dir='', 
        show_plots=False, 
        save_dcm_plot=False,
        save_fit_dirs=r"fits/", 
        manual_init_list=None, 
        plot_extra=False):
    """
    Fits multiple resonances at different powers for a given resonator.
    """

    if single_fit_results is not None:
        filenames = [r["filename"] for r in single_fit_results]
        powers = [r["power"] for r in single_fit_results]

    if len(filenames) != len(powers):
        raise ValueError(f"Length mismatch: {len(filenames)=}, {len(powers)=}")

    if manual_init_list is None:
        manual_init_list = [None] * len(filenames)

    Npts = len(filenames)
    navg = np.zeros(Npts)
    fc = np.zeros(Npts)
    fc_err = np.zeros(Npts)
    Qi = np.zeros(Npts)
    Qi_err = np.zeros(Npts)
    Qc = np.zeros(Npts)
    Qc_err = np.zeros(Npts)
    Q = np.zeros(Npts)
    Q_err =np.zeros(Npts)

    for idx, filename in enumerate(filenames):
        if single_fit_results is None:
            manual_init = manual_init_list[idx]
            params, err, conf_int, fig = fit_single_res(
                filename,
                preprocess_method=preprocess_method,
                save_dcm_plot=save_dcm_plot,
                save_fit_dirs=save_fit_dirs,
                manual_init=manual_init,
                plot_extra=plot_extra)
        else:
            r = single_fit_results[idx]
            params = r["params"]
            err = r["err"]
            conf_int = r["conf_int"]
            fig = r["fig"]


        Q[idx] = params[0]

        fscale = 1e9 if params[2] > 1e9 else 1
        fc[idx] = params[2] / fscale

        Qcj = params[1] * np.exp(1j * (params[3] + phi0))
        Qij = 1.0 / (1.0 / params[0] - np.real(1.0 / Qcj))
        
        Qi_val = np.real(Qij)
        Qc_val = np.real(Qcj)

        if not np.isfinite(Qi_val) or not np.isfinite(Qc_val) or Qi_val <= 0 or Qc_val <= 0:
            print(f"[WARNING] Bad fit at power {powers[idx]} dBm → skipping")
            Qi[idx] = np.nan
            Qc[idx] = np.nan
        else:
            Qi[idx] = Qi_val
            Qc[idx] = Qc_val
        
        # Intentionally use Qc[0] and fc[0] as reference values for photon-number conversion
        power_at_device = powers[idx] + np.sum(atten)

        navg_val = power_to_navg(power_at_device, Qi[idx], Qc[0], fc[0])
        navg[idx] = np.real(navg_val)

        Qi_err[idx] = conf_int[1]
        Qc_err[idx] = conf_int[2]
        fc_err[idx] = conf_int[5] / fscale
        Q_err[idx] = conf_int[0]

        print(f"navg: {navg[idx]:.0f} photons")
        print(f"Q: {Q[idx]:.0f} +/- {conf_int[0]:.0f}")
        print(f"Qi: {Qi[idx]:.0f} +/- {Qi_err[idx]:.0f}")
        print(f"Qc: {Qc[idx]:.0f} +/- {Qc_err[idx]:.0f}")
        print(f"fc: {fc[idx]:.9f} +/- {fc_err[idx]:.9f} GHz")
        print("-------------\n")

        if show_plots is False:
            plt.close('all')
        else:
            fig.show()

    df = pd.DataFrame(
        np.vstack((powers, navg, fc, fc_err, Qi, Qi_err, Qc, Qc_err, Q, Q_err)).T,
        columns=['Power [dBm]', 'navg', 'fc [GHz]', 'fc error', 'Qi', 'Qi error', 'Qc', 'Qc error', 'Q', 'Q error']
    )

    dstr = datetime.datetime.today().strftime('%y%m%d_%H_%M_%S')

    if data_dir:
        data_dir = Path(data_dir)
        folder_dir = data_dir.name
        filename_csv = f"qiqcfc_vs_power_{dstr}.csv"

        df.to_csv(data_dir / filename_csv, index=False)
    else:
        filename_csv = f"qiqcfc_vs_power_{dstr}.csv"
        df.to_csv(filename_csv, index=False)

    return df

def fit_delta_tls(
        Qi, T, fc, Qc, p, display_scales={'QHP': 1e6, 'nc': 1e1, 'Fdtls': 1e-6}, QHP_fix=False, Qierr=None,
        fit_init=None, fit_bounds=None):
    """
    Fit the TLS-related loss model:

    delta_tls = F * delta0_tls * tanh(hbar*w_c / 2kB*T) * (1 + <n>/nc)^(-beta)
                + 1/QHP

    Parameters
    ----------
    Qi : array-like
        Internal quality factor values.
    T : float
        Temperature. If T > 0.4, interpreted as mK and converted to K.
    fc : float
        Resonance frequency in GHz or Hz.
    Qc : float or array-like
        Coupling quality factor.
    p : array-like
        Applied power in dBm.
    display_scales : dict
        Reserved for display scaling.
    QHP_fix : bool
        If True, fix QHP to the maximum-Qi point.
    Qierr : array-like or None
        Qi uncertainties.
    fit_init : list or tuple or None
        Optional initial guess for curve_fit.
        If QHP_fix=True: [Fdtls, nc, beta]
        If QHP_fix=False: [Fdtls, nc, QHP, beta]
    fit_bounds : 2-tuple or None
        Optional bounds passed to scipy.optimize.curve_fit.
    """
    Qi = np.asarray(Qi, dtype=float)
    p = np.asarray(p, dtype=float)

    if Qierr is not None:
        Qierr = np.asarray(Qierr, dtype=float)

    # Physical constants
    h = 6.626069934e-34
    hbar = h / (2 * np.pi)
    kB = 1.3806485e-23

    # Unit handling
    fc_Hz = fc if np.any(np.asarray(fc) >= 1e9) else fc * 1e9
    TK = T if T <= 400e-3 else T * 1e-3

    delta = 1.0 / Qi
    hw0 = hbar * 2 * np.pi * fc_Hz
    kT = kB * TK

    navg = np.abs(power_to_navg(p, Qi, Qc, fc))
    labels = [r'$10^{%.2g}$' % x for x in np.log10(navg)]
    print(f'<n>: {labels}')
    print(f'T: {TK} K')
    print(f'fc_Hz: {fc_Hz} Hz')

    def fitfun_free(n, Fdtls, nc, QHP, beta):
        num = Fdtls * np.tanh(hw0 / (2 * kT))
        den = (1.0 + n / nc) ** beta
        return num / den + 1.0 / QHP

    if QHP_fix:
        if Qierr is None:
            raise ValueError("Qierr must be provided when QHP_fix=True")

        QHPidx = np.argmax(Qi)
        QHP = Qi[QHPidx]
        QHP_err = Qierr[QHPidx]

        def fitfun_fixed(n, Fdtls, nc, beta):
            num = Fdtls * np.tanh(hw0 / (2 * kT))
            den = (1.0 + n / nc) ** beta
            return num / den + 1.0 / QHP

        x0 = fit_init if fit_init is not None else [2.2e5, 1.0, 0.25]

        if fit_bounds is None:
            valid = np.isfinite(navg) & np.isfinite(delta)

            navg_fit = navg[valid]
            delta_fit_data = delta[valid]

            if len(navg_fit) < 4:
                raise ValueError(f"Not enough valid points for TLS fit: {len(navg_fit)}")

            popt, pcov = sp.optimize.curve_fit(fitfun_fixed, navg_fit, delta_fit_data, p0=x0)
            print(f"[TLS DEBUG] valid points: {len(navg_fit)} / {len(navg)}")
        else:
            popt, pcov = sp.optimize.curve_fit(
                fitfun_fixed, navg, delta, p0=x0, bounds=fit_bounds
            )

        Fdtls, nc, beta = popt
        errs = np.sqrt(np.diag(pcov))
        Fdtls_err, nc_err, beta_err = errs

    else:
        x0 = fit_init if fit_init is not None else [2.2e5, 1.0, np.max(Qi), 0.25]

        if fit_bounds is None:
            popt, pcov = sp.optimize.curve_fit(fitfun_free, navg, delta, p0=x0)
        else:
            popt, pcov = sp.optimize.curve_fit(
                fitfun_free, navg, delta, p0=x0, bounds=fit_bounds
            )

        Fdtls, nc, QHP, beta = popt
        errs = np.sqrt(np.diag(pcov))
        Fdtls_err, nc_err, QHP_err, beta_err = errs

    # Helper: round uncertainty to n significant figures
    def round_sigfig(x, n):
        if x == 0:
            return 0.0
        return round(x, n - int(np.floor(np.log10(abs(x)))) - 1)

    Fdtls_err = round_sigfig(Fdtls_err, 1)
    nc_err = round_sigfig(nc_err, 1)
    QHP_err = round_sigfig(QHP_err, 1)
    beta_err = round_sigfig(beta_err, 1)

    Fdtls_un = uncertainties.ufloat(Fdtls, Fdtls_err)
    nc_un = uncertainties.ufloat(nc, nc_err)
    QHP_un = uncertainties.ufloat(QHP, QHP_err)
    beta_un = uncertainties.ufloat(beta, beta_err)

    print(f'QHP: {QHP:.2f}+/-{QHP_err:.2f}')

    # Fdtls_latex = f'{Fdtls_un:.1uL}'
    # nc_latex    = f'{nc_un:.1uL}'
    # QHP_latex   = f'{QHP_un:.1uL}'
    # beta_latex  = f'{beta_un:.1uL}'

    # Fdtls_str = r'$F\delta^{0}_{TLS}: %s$' % Fdtls_latex
    # nc_str = r'$n_c: %s$' % nc_latex
    # QHP_str = r'$Q_{HP}: %s$' % QHP_latex
    # beta_str = r'$\beta: %s$' % beta_latex
    # delta_fit_str = Fdtls_str + '\n' + nc_str + '\n' + QHP_str + '\n' + beta_str

    Fdtls_latex = f'{Fdtls_un:L}'
    nc_latex    = f'{nc_un:L}'
    QHP_latex   = f'{QHP_un:L}'
    beta_latex  = f'{beta_un:L}'

    Fdtls_str = rf'$F\delta^0_{{TLS}}: {Fdtls_latex}$'
    nc_str    = rf'$n_c: {nc_latex}$'
    QHP_str   = rf'$Q_{{HP}}: {QHP_latex}$'
    beta_str  = rf'$\beta: {beta_latex}$'

    delta_fit_str = '\n'.join([Fdtls_str, nc_str, QHP_str, beta_str])

    print(delta_fit_str)

    if QHP_fix:
        return (
            Fdtls, nc, QHP,
            Fdtls_err, nc_err, QHP_err,
            fitfun_fixed(navg, *popt),
            delta_fit_str
        )
    else:
        return (
            Fdtls, nc, QHP,
            Fdtls_err, nc_err, QHP_err,
            fitfun_free(navg, *popt),
            delta_fit_str
        )

def power_to_navg(
        power_dBm, Qi, Qc, fc):
    """
    Converts power to photon number following Eq. (1) of arXiv:1801.10204 and Eq. (3) of arXiv:1912.09119
    """
    # Physical constants, Planck's constant J s
    h = 6.62607015e-34
    hbar = h/2/np.pi

    # Convert dBm to W
    Papp = 10**((power_dBm - 30) / 10) # * 1e-3
    fc_arr = np.asarray(fc, dtype=float)
    fscale = np.where(fc_arr > 1e9, 1.0, 1e9)
    fc_Hz = fc_arr * fscale
    hb_wc2 = hbar * (2 * np.pi * fc_Hz)**2

    # Return the power as average number of photons
    Q = 1. / ((1. / Qi) + (1. / Qc))
    Qi_arr = np.asarray(Qi, dtype=float)
    Qc_arr = np.asarray(Qc, dtype=float)

    bad = (Qi_arr <= 0) | (Qc_arr <= 0) | (~np.isfinite(Qi_arr)) | (~np.isfinite(Qc_arr))

    Q = 1. / ((1. / Qi_arr) + (1. / Qc_arr))
    navg = (2. / hb_wc2) * (Q**2 / np.abs(Qc_arr)) * Papp
    navg = np.where(bad, np.nan, navg)

    return np.real(navg)

def power_sweep_fit_drv(
        sample_name=None,
        temperature=0.010,      
        powers_in=None,
        all_paths=None,
        atten=[0, -70],
        save_fit_dirs="fits/",
        data_dir=None,
        
        plot_fit=True,
        plot_extra=False,
        save_dcm_plot=False,
        show_plots=False,

        use_error_bars=True,
        phi0=0,
        loss_scale=None,

        preprocess_method='circle',
        ds={'QHP': 1.0e6, 'nc': 1, 'Fdtls': 1e-5},
        plot_twinx=False,
        QHP_fix=True,
        manual_init_list=None,
        show_dbm=True,

        tls_fit_init=None, 
        tls_fit_bounds=None,):
    """
    Driver for fitting the power sweep data for a given set of data.
    Returns the fitted dataframe and figure handles.
    """

    # Validate input powers
    if powers_in is None or len(powers_in) == 0:
        raise ValueError("powers_in is None or empty.")
    powers = np.asarray(powers_in, dtype=float)

    # Validate input file paths
    if all_paths is None or len(all_paths) == 0:
        raise ValueError("all_paths is None or empty.")
    filenames = list(all_paths)

    if len(filenames) != len(powers):
        raise ValueError(f"Mismatch between files and powers: {len(filenames)=}, {len(powers)=}")

    print(f"powers: {powers}")

    dstr = datetime.datetime.today().strftime('%y_%m_%d')
    fsize = 20
    csize = 5

    # Perform resonator fits
    single_fit_results = fit_single_power_list(
        filenames,
        powers,
        preprocess_method=preprocess_method,
        phi0=phi0,
        show_plots=show_plots,
        save_dcm_plot=True,
        save_fit_dirs=save_fit_dirs,
        manual_init_list=manual_init_list,
        plot_extra=plot_extra
    )

    df = fit_qiqcfc_vs_power(
        single_fit_results=single_fit_results,
        atten=atten,
        preprocess_method=preprocess_method,
        phi0=phi0,
        data_dir=data_dir,
        show_plots=show_plots,
        save_dcm_plot=save_dcm_plot,
        save_fit_dirs=save_fit_dirs,
        manual_init_list=manual_init_list,
        plot_extra=plot_extra
    )

    # Extract fit results
    navg = np.asarray(df['navg'])
    fc = np.asarray(df['fc [GHz]'])
    fc_err = np.asarray(df['fc error'])
    Qi = np.asarray(df['Qi'])
    Qi_err = np.asarray(df['Qi error'])
    Qc = np.asarray(df['Qc'])
    Qc_err = np.asarray(df['Qc error'])
    Q = np.asarray(df['Q'])
    Q_err = np.asarray(df['Q error'])

    delta = 1.0 / Qi
    delta_err = Qi_err / Qi**2

    # Add attenuation to powers
    powers_total = powers + sum(atten)

    def pdBm_to_navg_ticks(p):
        n = np.abs(power_to_navg(powers_total[0::2], Qi[0::2], Qc[0], fc[0]))
        labels = [r'$10^{%.2g}$' % x for x in np.log10(n)]
        print(f'labels:\n{labels}')
        return labels

    # TLS loss fit
    T = temperature
    doff = 0
    delta_fit = None
    delta_fit_str = None
    
    if plot_fit and len(Qi) >= 4:
        fit_out = fit_delta_tls(
                Qi, T, fc[0], Qc[0], powers_total,
                display_scales=ds,
                QHP_fix=QHP_fix,
                Qierr=Qi_err,
                fit_init=tls_fit_init,
                fit_bounds=tls_fit_bounds)

        Fdtls, nc, QHP, Fdtls_err, nc_err, QHP_err, delta_fit, delta_fit_str = fit_out

        if loss_scale:
            delta_fit = delta_fit / loss_scale

        print()
        print(f'F * d0_tls: {Fdtls:.2g} +/- {Fdtls_err:.2g}')
        print(f'nc: {nc:.2g} +/- {nc_err:.2g}')
        print()

    else:
        print("[INFO] Skipping TLS fit (not enough power points)")

    if loss_scale:
        delta /= loss_scale
        delta_err /= loss_scale

    # Create figures
    fig_fc, ax_fc = plt.subplots(1, 1, tight_layout=True)
    ax_fc.set_xlabel('Power [dBm]', fontsize=fsize)
    ax_fc.set_ylabel('Res Freq Shift From High Power [GHz]', fontsize=fsize)
    ax_fc_top = ax_fc.twiny()

    plot_kwargs = {"figsize": (8, 6)}
    fig_qc, ax_qc = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
    fig_qi, ax_qi = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
    fig_qiqc, ax_qiqc = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)
    fig_d, ax_d = plt.subplots(1, 1, tight_layout=True, **plot_kwargs)

    xvals = powers_total.copy()
    if not plot_twinx:
        xvals = np.abs(power_to_navg(powers_total, Qi, Qc[0], fc[0]))

    # Plot data
    if use_error_bars:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax_fc.errorbar(xvals, fc - fc[0], yerr=fc_err, marker='o', ls='', ms=10, capsize=csize)
        ax_qc.errorbar(xvals, Qc, yerr=Qc_err, marker='o', ls='', ms=10, capsize=csize)
        ax_qiqc.errorbar(xvals, Qi, yerr=Qi_err, marker='h', ls='', ms=10,
                         capsize=csize, color=colors[5], label=r'$Q_i$')
        ax_qiqc.errorbar(xvals, Qc, yerr=Qc_err, marker='^', ls='', ms=10,
                         capsize=csize, color=colors[6], label=r'$Q_c$')
        ax_qi.errorbar(xvals, Qi, yerr=Qi_err, marker='o', ls='', ms=10, capsize=csize)

        if doff > 0:
            ax_d.errorbar(xvals[0:-doff], delta[0:-doff], yerr=delta_err[0:-doff],
                          marker='d', ls='', color=colors[1], ms=10, capsize=csize)
            if plot_fit and delta_fit is not None:
                ax_d.plot(xvals[0:-doff], delta_fit, ls='-', label=delta_fit_str, color=colors[1])
        else:
            ax_d.errorbar(xvals, delta, yerr=delta_err, marker='d', ls='',
                          color=colors[1], ms=10, capsize=csize)
            if plot_fit and delta_fit is not None:
                ax_d.plot(xvals, delta_fit, ls='-', label=delta_fit_str, color=colors[1])

    else:
        ax_fc.plot(xvals, fc - fc[0], marker='o', ms=10, ls='')
        ax_qc.plot(xvals, Qc, marker='o', ms=10, ls='')
        ax_qi.plot(xvals, Qi, marker='o', ms=10, ls='')
        ax_d.plot(xvals, delta, marker='o', ms=10, ls='')

    if show_dbm:
        for x, y, text in zip(xvals, delta, powers_in):
            ax_d.text(x, y, f"{text} dBm", size=12, rotation=45,
                      rotation_mode="anchor", horizontalalignment="left",
                      verticalalignment="bottom")

    ax_qc.set_ylabel(r'$Q_c$', fontsize=fsize)
    ax_qi.set_ylabel(r'$Q_i$', fontsize=fsize)
    ax_qiqc.set_ylabel(r'$Q_i, Q_c$', fontsize=fsize)

    if loss_scale:
        ax_d.set_ylabel(r'$Q_i^{-1}\times 10^{%d}$' % int(np.log10(loss_scale)), fontsize=fsize)
    else:
        ax_d.set_ylabel(r'$Q_i^{-1}$', fontsize=fsize)

    power_str = f'{atten[0]} dB ext, {atten[1]} dB int attenuation'
    for ax in [ax_fc, ax_qc, ax_qi, ax_qiqc, ax_d]:
        ax.set_title(power_str, fontsize=fsize)

    # Top axes
    if plot_twinx:
        ax_d_top = ax_d.twiny()
        ax_qc_top = ax_qc.twiny()
        ax_qi_top = ax_qi.twiny()
        ax_qiqc_top = ax_qiqc.twiny()

        ax_qc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_qi.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_qiqc.set_xlabel('Power [dBm]', fontsize=fsize)
        ax_d.set_xlabel('Power [dBm]', fontsize=fsize)

        ax_qc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qi_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qiqc_top.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
        ax_d_top.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)
        ax_fc_top.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)

        for ax in [ax_qc, ax_qi, ax_qiqc, ax_d, ax_fc]:
            ax.set_xticks(xvals[0::2])

        ax_qc_top.set_xticks(ax_qc.get_xticks())
        ax_qi_top.set_xticks(ax_qi.get_xticks())
        ax_qiqc_top.set_xticks(ax_qiqc.get_xticks())
        ax_d_top.set_xticks(ax_d.get_xticks())
        ax_fc_top.set_xticks(ax_fc.get_xticks())

        ax_qc_top.set_xbound(ax_qc.get_xbound())
        ax_qi_top.set_xbound(ax_qi.get_xbound())
        ax_qiqc_top.set_xbound(ax_qiqc.get_xbound())
        ax_d_top.set_xbound(ax_d.get_xbound())
        ax_fc_top.set_xbound(ax_fc.get_xbound())

        ax_qc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qc.get_xticks()))
        ax_qi_top.set_xticklabels(pdBm_to_navg_ticks(ax_qi.get_xticks()))
        ax_qiqc_top.set_xticklabels(pdBm_to_navg_ticks(ax_qiqc.get_xticks()))
        ax_d_top.set_xticklabels(pdBm_to_navg_ticks(ax_d.get_xticks()))
        ax_fc_top.set_xticklabels(pdBm_to_navg_ticks(ax_fc.get_xticks()))

        set_xaxis_rot(ax_d_top, 45)
        set_xaxis_rot(ax_d, 45)

    else:
        for ax in [ax_qc, ax_qi, ax_qiqc, ax_fc, ax_d]:
            ax.set_xscale('log')

        ax_qi.set_yscale('log')
        ax_qiqc.set_yscale('log')

        ax_qc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qi.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_qiqc.set_xlabel(r'[$\left<{n}\right>$]', fontsize=fsize)
        ax_fc.set_xlabel(r'Power [$\left<{n}\right>$]', fontsize=fsize)
        ax_d.set_xlabel(r'$\left<{n}\right>$', fontsize=fsize)

    # Legends
    qiqc_handles, qiqc_labels = ax_qiqc.get_legend_handles_labels()
    if qiqc_handles:
        ax_qiqc.legend(qiqc_handles, qiqc_labels, fontsize=fsize)

    d_handles, d_labels = ax_d.get_legend_handles_labels()
    if d_handles:
        ax_d.legend(d_handles, d_labels, loc='upper right', fontsize=fsize)

    # Naming
    try:
        fc_val = hm.get_frequency_from_filename(str(filenames[0]))
    except Exception:
        print(f"{filenames[0] = }")
        fc_val = 0.1

    fc_str = f"{fc_val:1.3f}".replace(".", "p")
    fsuffix = f"_{fc_str}GHz_{temperature}mK_{dstr}.png"

    fig_fc_title = 'fc_vs_power' + fsuffix
    fig_qc_title = 'qc_vs_power' + fsuffix
    fig_qiqc_title = 'qiqc_vs_power' + fsuffix
    fig_qi_title = 'qi_vs_power' + fsuffix
    fig_d_title = 'tand_vs_power' + fsuffix

    fig_title_size = 24
    fig_fc.suptitle(fig_fc_title.replace(".png", ""), fontsize=fig_title_size)
    fig_qc.suptitle(fig_qc_title.replace(".png", ""), fontsize=fig_title_size)
    fig_qiqc.suptitle(fig_qiqc_title.replace(".png", ""), fontsize=fig_title_size)
    fig_qi.suptitle(fig_qi_title.replace(".png", ""), fontsize=fig_title_size)
    fig_d.suptitle(fig_d_title.replace(".png", ""), fontsize=fig_title_size)

    for fig in [fig_fc, fig_qc, fig_qiqc, fig_qi, fig_d]:
        fig.tight_layout()

    # Save figures
    save_dirs = []
    if data_dir is not None:
        data_dir = Path(data_dir)
        plot_dir = data_dir / "all_fit_plots"
        # report_dir = Path("reports") / f"{sample_name}_{fc_str}GHz"
        hm.check_and_make_dir(str(plot_dir))
        # hm.check_and_make_dir(str(report_dir))
        # save_dirs.extend([plot_dir, report_dir])
        save_dirs.extend([plot_dir])
    else:
        local_dir = Path.cwd() / "all_fit_plots"
        hm.check_and_make_dir(str(local_dir))
        save_dirs.append(local_dir)

    for out_dir in save_dirs:
        fig_fc.savefig(out_dir / fig_fc_title, format='png')
        fig_qc.savefig(out_dir / fig_qc_title, format='png')
        fig_qiqc.savefig(out_dir / fig_qiqc_title, format='png')
        fig_qi.savefig(out_dir / fig_qi_title, format='png')
        fig_d.savefig(out_dir / fig_d_title, format='png')

    if show_plots is False:
        plt.close('all')

    return {
        "df": df,
        "fig_fc": fig_fc,
        "fig_qc": fig_qc,
        "fig_qiqc": fig_qiqc,
        "fig_qi": fig_qi,
        "fig_d": fig_d,
    }

def set_xaxis_rot(
        ax, angle=45.):
    """
    Rotate x-axis labels
    """
    for tick in ax.get_xticklabels():
        tick.set_rotation(angle)

