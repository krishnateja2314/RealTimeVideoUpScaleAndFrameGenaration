"""
Application configuration and constants.
"""

from pathlib import Path
import platform

# Platform detection
OS_TYPE = platform.system()  # 'Linux', 'Windows', 'Darwin'

# Application paths
APP_ROOT = Path(__file__).parent
MODELS_DIR = APP_ROOT / "models"
CONFIG_DIR = Path.home() / ".upscaler_config"
CACHE_DIR = CONFIG_DIR / "cache"

# ONNX Model filenames
INTERPOLATOR_MODEL = "frame_interpolator.onnx"
UPSCALER_MODEL = "espcn_4x_dynamic.onnx"

# Performance estimation defaults
PERF_ESTIMATION_FRAMES = 5
PERF_OVERHEAD_MARGIN = 1.2  # 20% overhead

# Batch size detection
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 128
BATCH_SIZE_TEST_MULTIPLIER = 2

# Video processing
DEFAULT_OUTPUT_CODEC = 'mp4v'
SUPPORTED_CODECS = ['mp4v', 'h264', 'h265', 'vp9']
DEFAULT_TARGET_FPS = 60
DEFAULT_INTERPOLATION = 1

# Frame padding multiple (ONNX model requirement)
FRAME_PADDING_MULTIPLE = 32

# GPU memory thresholds
GPU_MEMORY_RESERVE_MB = 512  # Reserve this much GPU memory

# Logging
LOG_DIR = CONFIG_DIR / "logs"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'

# Ensure directories exist
for dir_path in [CONFIG_DIR, CACHE_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
