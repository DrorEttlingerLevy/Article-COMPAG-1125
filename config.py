from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATE_PATH = DATA_DIR / "template" / "exp_1_karnel.parquet"
OUTPUT_DIR = DATA_DIR / "outputs"

EXPECTED_SAMPLE_RATE = 48_000
EXPECTED_CHANNELS = 6
CHANNEL_IDS = [0, 1, 2, 3, 4, 5]

# Processing
CHUNK_SECONDS = 60
SPECTRAL_GATE_THRESHOLD = 1.5
AVERAGE_FACTOR = 1000
ROUND_DECIMALS = 8

# Sliding window on the averaged signal (48 Hz after averaging factor 1000)
WINDOW_SECONDS = 10 * 60
STEP_SECONDS = 6

# Noise reduction configuration
NOISE_REDUCE_STATIONARY = True
NOISE_REDUCE_PROP_DECREASE = 1.0

# Output
SAVE_PLOTS = True
PLOT_DPI = 120
