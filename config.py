# General FL parameters
NUM_ROUNDS = 10
EPOCHS_PER_CLIENT = 3
BATCH_SIZE = 1
START_ROUND = 1
CURRENT_ROUND=1
LEARNING_RATE = 1e-4

CLIENT_DATA_SIZES = {"train": 56, "val": 13, "test": 12}

# Paths
BASE_DIR = "./"
SHARED_DIR = "shared"  # for model exchange between clients and server

# Hardware
DEVICE = "cuda"  # or "cpu"
