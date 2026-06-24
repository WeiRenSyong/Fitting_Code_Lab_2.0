# %%
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR / "helper_scripts")) # custom module folder 1
sys.path.insert(0, str(BASE_DIR / "scresonators")) # custom module folder 2

import helper_fit as hf
import helper_misc as hm
# import regex as re
import re
import matplotlib.pyplot as plt

print("helper_fit   ->", hf.__file__)
print("helper_misc  ->", hm.__file__)

# %%  Get sample name and device information
# Define data directory
data_dir = Path(r"C:\Users\user\OneDrive\文件\GitHub\Measurements\Cooldown_78_Line6-QSD_CPW_w50g25_05")

# Safety check
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory not found: {data_dir}")

current_dir = data_dir

# Parse folder name
parts = re.split(r"-", current_dir.name, maxsplit=1)

if len(parts) < 2:
    raise ValueError(f"Folder name format invalid: {current_dir.name}")

line_num, sample_name = parts

# Debug print
print("=== Sample Info ===")
print(f"Data directory : {current_dir}")
print(f"Line number    : {line_num}")
print(f"Sample name    : {sample_name}")

# %% Define the search folder
folder_name = 'most_recent_data'
search_folder = current_dir / folder_name

# Safety check
if not search_folder.exists():
    raise FileNotFoundError(f"Search folder not found: {search_folder}")

# Only keep resonator folders
pattern = re.compile(r"Resonator_\d+_.*GHz")

chosen_resonators = [
    x for x in search_folder.iterdir()
    if x.is_dir() and pattern.match(x.name)
]

# Sort for reproducibility
chosen_resonators = sorted(chosen_resonators)

# Debug print
print("=== Resonator Folders ===")
print(f"Search folder: {search_folder}")
print(f"Found {len(chosen_resonators)} resonators:")

for r in chosen_resonators:
    print(f"  - {r.name}")


# %% perform power sweep

TLS_FIT_CONFIGS = {
    "MQC_Ta_": {
        "tls_fit_init": [2.2e5, 1.0, 0.25],   # QHP_fix=True case
        "tls_fit_bounds": None,
    },
    "default": {
        "tls_fit_init": None,
        "tls_fit_bounds": None,
    }
}

tls_cfg = TLS_FIT_CONFIGS.get(sample_name, TLS_FIT_CONFIGS["default"])

external_attenuation = 0
internal_attenuation = -70
temperature_mK = 205

# manual_guess_res = [1640, 5830, 7.211780e9, -0.02] # [Q, Qc, f_c, phi]

for resonator_path in chosen_resonators:
    print("\n==============================")
    print(f"Processing resonator: {resonator_path.name}")

    all_resonator_csvs_paths = sorted(
        [x for x in resonator_path.glob("*GHz*.csv") if "dBm" in x.name]
    )
    all_resonator_csvs_names = [x.name for x in all_resonator_csvs_paths]

    if len(all_resonator_csvs_paths) == 0:
        print(f"[WARNING] No valid CSV files found in {resonator_path}")
        continue

    all_powers = [hm.get_power_from_filename(fname) for fname in all_resonator_csvs_names]

    if len(all_resonator_csvs_paths) != len(all_powers):
        raise ValueError(f"Mismatch between CSV files and extracted powers in {resonator_path}")

    init_conds = [None] * len(all_resonator_csvs_paths)

    # for i, path in enumerate(all_resonator_csvs_paths):
    #     if "7p212GHz" in path.name:
    #         init_conds[i] = manual_guess_res

    save_fit_dirs = [str(resonator_path), str(resonator_path)]

    print(f"Number of files: {len(all_resonator_csvs_paths)}")
    for fname, pwr in zip(all_resonator_csvs_names, all_powers):
        print(f"  {fname} --> {pwr} dBm")
    
    try:
        hf.power_sweep_fit_drv(
            sample_name=sample_name,
            temperature=temperature_mK,
            powers_in=all_powers,
            all_paths=all_resonator_csvs_paths,
            atten=[external_attenuation, internal_attenuation],
            save_fit_dirs=save_fit_dirs,
            data_dir=resonator_path,

            plot_fit=True,
            plot_extra=False,
            save_dcm_plot=False,
            show_plots=False,

            use_error_bars=True,
            phi0=0.,
            loss_scale=1e-6,

            preprocess_method='circle',
            # preprocess_method=None,
            ds={'QHP': 1e5, 'nc': 1, 'Fdtls': 1e-4},
            plot_twinx=False,
            QHP_fix=True,   # Set QHP as the highest power Q first, which would make more accurate for S-curve fitting
            manual_init_list=init_conds,
            show_dbm=True,

            tls_fit_init=tls_cfg["tls_fit_init"],
            tls_fit_bounds=tls_cfg["tls_fit_bounds"],
        )
    except Exception as e:
        print(f"[ERROR] Failed on {resonator_path.name}: {e}")
    finally:
        plt.close('all')

print(f'Analyzing is done.')