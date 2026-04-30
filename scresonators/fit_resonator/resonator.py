"""DCM-only resonator container and data-loading utilities."""

import datetime
import os

import attr
import attrs
import numpy as np

import fit_resonator.cavity_functions as ff
import fit_resonator.fit as fit


class FitMethod:
    """
    Container for DCM fitting configuration.

    Parameters
    ----------
    method : str
        Must be "DCM" in this lab fork.
    MC_iteration : int
        Number of fit / Monte Carlo iterations.
    MC_rounds : int
        Number of random Monte Carlo trials per iteration.
    MC_weight : str
        'no' or 'yes'. If 'yes', use 1/|S21| weighting.
    MC_weightvalue : int
        Retained for compatibility.
    MC_fix : list[str] or None
        Parameters to hold fixed. Relevant DCM names are:
        'Q', 'Qc', 'w1', 'phi'
    MC_step_const : float
        Random step size used in Monte Carlo search.
    manual_init : list[float] or None
        Manual initial guess for DCM:
        [Q, Qc, f_c, phi]
    vary : list[bool] or None
        Per-parameter vary mask.
    preprocess_method : str
        "linear" or "circle"
    """

    def __init__(self,
                 method: str = "DCM",
                 MC_iteration: int = 5,
                 MC_rounds: int = 100,
                 MC_weight: str = 'no',
                 MC_weightvalue: int = 2,
                 MC_fix=None,
                 MC_step_const: float = 0.6,
                 manual_init=None,
                 vary=None,
                 preprocess_method: str = "linear"):

        if method != "DCM":
            raise ValueError("This lab fork currently supports DCM only.")

        if manual_init is not None:
            if not isinstance(manual_init, list) or len(manual_init) != 4:
                raise ValueError("manual_init must be None or a list of length 4: [Q, Qc, f_c, phi]")

        if MC_fix is None:
            MC_fix = []

        self.method = "DCM"
        self.func = ff.cavity_DCM
        self.MC_rounds = MC_rounds
        self.MC_iteration = MC_iteration
        self.MC_weight = MC_weight
        self.MC_weightvalue = MC_weightvalue
        self.MC_step_const = MC_step_const
        self.MC_fix = MC_fix
        self.manual_init = manual_init
        self.vary = vary if vary is not None else [True] * 6
        self.preprocess_method = preprocess_method

    def __repr__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())

    def change_method(self, method: str):
        if method != "DCM":
            raise ValueError("This lab fork currently supports DCM only.")
        if self.method == method:
            print("Fit method does not change")
        else:
            self.method = "DCM"
            self.func = ff.cavity_DCM


class ResonatorData:
    """Simple container for resonator data."""
    def __init__(self,
                 freqs: np.ndarray,
                 amps: np.ndarray,
                 phases: np.ndarray,
                 linear_amps: np.ndarray):
        self.freqs = freqs
        self.amps = amps
        self.phases = phases
        self.linear_amps = linear_amps


@attr.define(init=True)
class Resonator:
    """
    Object representing a DCM resonator fit.
    """

    filepath: str = None
    data: ResonatorData = None
    databg: ResonatorData = None
    method_class: FitMethod = None
    name: str = ''
    date: datetime.datetime = None
    temp: float = None
    bias: float = None
    measurement: str | int | tuple | list | np.ndarray = None
    normalize: int = 10
    background: str = None
    background_array: np.ndarray = None
    plot: str = 'pdf'
    plot_extra: bool = False
    preprocess_method: str = "circle"
    save_dcm_plot: bool = False
    fit_dir: str = ".\\"
    power: float = 0

    def __attrs_post_init__(self):
        if self.filepath is not None and self.data is None:
            self.from_file()

        if self.data is not None and not isinstance(self.data, ResonatorData):
            self.from_columns(self.data.T[0], self.data.T[1], self.data.T[2])

        if self.background is not None and self.databg is None:
            self.init_background(filepath=self.background)

        if self.background_array is not None and self.databg is None:
            self.init_background_array(self.background_array)

    def init_background(self, filepath=None, fscale=1):
        if self.background is None and filepath is not None:
            self.background = filepath
        self.databg = from_file(self.background, fscale=fscale)

    def init_background_array(self, bg_array=None):
        if self.background_array is None and bg_array is not None:
            self.background_array = bg_array
        self.databg = from_columns(
            self.background_array.T[0],
            self.background_array.T[1],
            self.background_array.T[2]
        )

    def from_columns(self, freqs, amps=None, phases=None):
        if freqs is not None and amps is None and phases is None:
            self.data = from_columns(freqs.T[0], freqs.T[1], freqs.T[2])
        else:
            self.data = from_columns(freqs, amps, phases)

    def from_file(self, filepath=None, measurement=None, fscale=1):
        if self.filepath is None and filepath is not None:
            self.filepath = filepath
        if self.measurement is None and measurement is not None:
            self.measurement = measurement
        self.data = from_file(self.filepath, data_column=measurement, fscale=fscale)

    def fit_method(self,
                   method: str = "DCM",
                   MC_iteration: int = 5,
                   MC_rounds: int = 100,
                   MC_weight: str = 'no',
                   MC_weightvalue: int = 2,
                   MC_fix=None,
                   MC_step_const: float = 0.6,
                   manual_init=None,
                   vary=None,
                   preprocess_method: str = None):

        if preprocess_method is None:
            preprocess_method = self.preprocess_method
        if MC_fix is None:
            MC_fix = []

        self.method_class = FitMethod(
            method=method,
            MC_iteration=MC_iteration,
            MC_rounds=MC_rounds,
            MC_weight=MC_weight,
            MC_weightvalue=MC_weightvalue,
            MC_fix=MC_fix,
            MC_step_const=MC_step_const,
            manual_init=manual_init,
            vary=vary,
            preprocess_method=preprocess_method
        )

    def fit(self, plot: str = 'pdf'):
        self.plot = plot
        return fit.fit(self)

    def load_params(self, method: str, params: np.ndarray, chi):
        """
        Load DCM fit parameters.
        """
        if method != "DCM":
            raise ValueError("This lab fork currently supports DCM only.")

        if not hasattr(self, "method") or self.method is None:
            self.method = []

        self.fc = params[2]
        if method not in self.method:
            self.method.append(method)
            self.DCMparams = DCMparams(params, chi)
            self.compare = ff.fit_raw_compare(
                self.data.freqs,
                self.data.amps,
                self.DCMparams.all,
                method
            )
        else:
            print("repeated load parameter")

    def reload_params(self, method: str, params: np.ndarray, chi):
        """
        Reload DCM fit parameters.
        """
        if method != "DCM":
            raise ValueError("This lab fork currently supports DCM only.")

        if hasattr(self, "method") and method in self.method:
            print(self.name + ' changed params')
            self.fc = params[2]
            self.DCMparams = DCMparams(params, chi)
            self.compare = ff.fit_raw_compare(
                self.data.freqs,
                self.data.amps,
                self.DCMparams.all,
                'DCM'
            )
        else:
            print('no')

    def power_calibrate(self, p):
        """
        Optional linear calibration of applied power.
        p = [a, b, c] for corrected_power = a*f + b*x + c + x
        """
        if not hasattr(self, "DCMparams"):
            raise ValueError("Please load DCM parameters first.")

        p = np.array(p)
        x = self.power
        f = self.fc
        self.corrected_power = p[0] * f + p[1] * x + p[2] + x

        hbar = 1.05e-34
        f_rad = 2 * np.pi * f * 1e9
        p_watts = 10 ** (self.corrected_power / 10 - 3)

        Q = self.DCMparams.Q
        Qc = self.DCMparams.Qc
        self.DCMparams.num_photon = 2 * p_watts / hbar / f_rad**2 * Q**2 / Qc


def from_columns(freqs, amps, phases):
    """Load data from columns provided individually."""
    linear_amps = 10 ** (amps / 20)
    return ResonatorData(freqs=freqs, amps=amps, phases=phases, linear_amps=linear_amps)


def from_file(filepath, data_column=None, fscale=1):
    """
    Load data from SNP, TXT, or CSV file.
    Frequencies are returned in GHz-scale convention used by the existing code.
    Phases are returned in radians.
    """
    if data_column is not None:
        s_col = data_column
    else:
        s_col = 1

    filename, extension = os.path.splitext(filepath)

    if extension.startswith('.s') and extension.endswith('p'):
        try:
            with open(filepath, 'r') as snp_file:
                file, inline, options, frequency_units, data_format = header_parse(file=snp_file)
                freqs, amps, phases, linear_amps = data_parse(
                    s_col, inline, frequency_units, data_format, file, options
                )
        except OSError as e:
            raise OSError(f"ERROR {e} when opening file: {filepath}")

        if frequency_units == 'hz':
            fscale = fscale / 1e9
        elif frequency_units == 'khz':
            fscale = fscale / 1e6
        elif frequency_units == 'mhz':
            fscale = fscale / 1e3

        freqs = freqs / fscale
        return ResonatorData(freqs, amps, phases, linear_amps)

    elif 'txt' in extension or 'csv' in extension:
        try:
            with open(filepath, 'r') as txt_file:
                file, line, options, frequency_units, data_format = header_parse(file=txt_file)
                data_lines = []
                while line:
                    if 'END' in line:
                        break
                    data_lines.append(line)
                    line = file.readline().strip()

            data = np.loadtxt(data_lines, delimiter=',')
        except Exception as e:
            raise ValueError(
                f"Exception '{e}' encountered when loading TXT/CSV file {filepath}. "
                f"Check delimiter/header formatting."
            )

        freqs = data.T[0] / fscale
        amps = data.T[1]
        phases = data.T[2] * np.pi / 180
        linear_amps = 10 ** (amps / 20)
        return ResonatorData(freqs, amps, phases, linear_amps)

    else:
        raise ValueError(
            f"File extension {extension} not supported. Please use .s2p, .s1p, .txt, or .csv"
        )


def header_parse(file):
    data_format = None
    frequency_units = None
    comment_line = ['!']
    option_line = ['#']
    metadata = ['s21', 's11', 's12', 's22']
    dformats = ['db', 'ma', 'ri']
    nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.']

    options = []
    inline = file.readline()

    while any(comment in inline.lower() for comment in comment_line) \
            or any(option_lead in inline for option_lead in option_line) \
            or not any(number in inline.lower()[0] for number in nums):
        if any(option_lead in inline.lower() for option_lead in option_line) \
                or any(dformat in inline.lower() for dformat in dformats) \
                or any(measure in inline.lower() for measure in metadata):
            options.append(inline)

        inline = file.readline()

    for val in options:
        if 'db' in val.lower():
            data_format = 'db'
        elif 'ma' in val.lower():
            data_format = 'ma'
        elif 'ri' in val.lower():
            data_format = 'ri'

        if 'ghz' in val.lower():
            frequency_units = 'ghz'
        elif 'khz' in val.lower():
            frequency_units = 'khz'
        elif 'mhz' in val.lower():
            frequency_units = 'mhz'
        elif 'hz' in val.lower():
            frequency_units = 'hz'

    return file, inline, options, frequency_units, data_format


def data_parse(s_col, line, frequency_units, data_format, file, options):
    row = line.split()
    data_rows = [3, 4]

    if len(row) == 0:
        raise ValueError("Data not found in file.")

    if len(row) > 3:
        if isinstance(s_col, int):
            data_rows[0] = s_col
        elif isinstance(s_col, (tuple, list, np.ndarray)):
            data_rows[0] = s_col[0]
            data_rows[1] = s_col[1]
        elif isinstance(s_col, str):
            for metadata in options:
                if 'Measurements: ' in metadata:
                    measurements = metadata.rsplit('Measurements: ')[1].strip('.:\n').lower().split(', ')
                    idx = (measurements.index(s_col.lower()) * 2) + 1
                    data_rows[0] = idx
                    data_rows[1] = idx + 1
                    break
        else:
            print("Could not interpret which data columns to use, using default")

    freqs = np.array(float(row[0]))

    if data_format == "db":
        amps = np.array(float(row[data_rows[0]]))
        phases = np.array(float(row[data_rows[1]]))
        line = file.readline().strip()

        while line:
            row = line.split()
            freqs = np.append(freqs, float(row[0]))
            amps = np.append(amps, float(row[data_rows[0]]))
            phases = np.append(phases, float(row[data_rows[1]]))
            line = file.readline().strip()

        phases = phases * np.pi / 180
        linear_amps = 10 ** (amps / 20)

    elif data_format == "ma":
        linear_amps = np.array(float(row[data_rows[0]]))
        phases = np.array(float(row[data_rows[1]]))
        line = file.readline().strip()

        while line:
            row = line.split()
            freqs = np.append(freqs, float(row[0]))
            linear_amps = np.append(linear_amps, float(row[data_rows[0]]))
            phases = np.append(phases, float(row[data_rows[1]]))
            line = file.readline().strip()

        phases = phases * np.pi / 180
        amps = np.log10(linear_amps) * 20

    elif data_format == "ri":
        real = np.array(float(row[data_rows[0]]))
        imaginary = np.array(float(row[data_rows[1]]))
        line = file.readline().strip()

        while line:
            row = line.split()
            freqs = np.append(freqs, float(row[0]))
            real = np.append(real, float(row[data_rows[0]]))
            imaginary = np.append(imaginary, float(row[data_rows[1]]))
            line = file.readline().strip()

        z = real + 1j * imaginary
        linear_amps = np.abs(z)
        phases = np.angle(z)   # radians
        amps = np.log10(linear_amps) * 20

    else:
        raise ValueError("Data type in file not supported. Please use DB, MA, or RI.")

    if frequency_units == "hz":
        freqs = freqs / 1e9
    elif frequency_units == "khz":
        freqs = freqs / 1e6
    elif frequency_units == "mhz":
        freqs = freqs / 1e3
    elif frequency_units != "ghz":
        print("Units for the frequency not found. Please include frequency units in the file header.")

    return freqs, amps, phases, linear_amps


@attrs.define
class DCMparams:
    """Container for DCM fitting results."""
    params: np.ndarray
    chi: float
    num_photon: float = 0

    def __attrs_post_init__(self):
        self.Q = self.params[0]
        self.Qc = self.params[1]
        self.fc = self.params[2]
        self.phi = ((self.params[3] + np.pi) % (2 * np.pi) - np.pi) / np.pi * 180
        self.all = self.params

        Qc_complex = self.params[1] * np.exp(1j * self.params[3])
        self.Qi = (self.params[0] ** -1 - np.real(Qc_complex ** -1)) ** -1
        self.ReQc = 1 / np.real(Qc_complex ** -1)