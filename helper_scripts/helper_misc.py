'''
    helper_misc.py
'''

from pathlib import Path
import regex as re

def check_and_make_dir(directory_name):
    """
    Create directory if it does not exist.
    Accepts str or Path. If a file-like path is passed, uses its parent folder.
    Returns the directory Path object.
    """
    path = Path(directory_name)

    # If user passed something like ".../file.csv", create the parent directory instead
    if path.suffix in ['.csv', '.pdf', '.png', '.txt']:
        path = path.parent

    if not path.exists():
        print(f'      target directory: {path}')
        print(f'      directory does not exist. now creating...')
        path.mkdir(parents=True, exist_ok=True)
    else:
        print(f'      directory already exists.')

    print(f'      absolute path: {path.resolve()}\n')
    return path
     
def get_power_from_filename(filename):
    """
    Extract power in dBm from filename.

    Supports patterns like:
    - 25dB   -> returns -25.0  (legacy behavior)
    - -25dBm -> returns -25.0
    - 25dBm  -> returns 25.0
    """
    filename = str(filename)

    # First try explicit dBm pattern
    match_dbm = re.search(r'(-?\d+(?:\.\d+)?)dBm', filename)
    if match_dbm:
        return float(match_dbm.group(1))

    # Fallback to legacy pattern like "25dB" meaning -25 dBm
    match_db = re.search(r'(\d{1,3}(?:\.\d+)?)dB', filename)
    if match_db:
        return -float(match_db.group(1))

    raise ValueError(f"Could not extract power from filename: {filename}")

def get_frequency_from_filename(filename):
    """
    Extract resonator frequency in GHz from filename.

    Example:
    - '_5p932GHz_' -> 5.932
    - '_12p3456GHz_' -> 12.3456
    """
    filename = str(filename)
    match = re.search(r'_(\d+)p(\d{1,6})GHz_', filename)

    if not match:
        raise ValueError(f"Could not extract frequency from filename: {filename}")

    whole = match.group(1)
    frac = match.group(2)
    return float(f"{whole}.{frac}")