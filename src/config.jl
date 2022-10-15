module Config
using Flux
include("default_config.jl") # Load default configuration, to be overwritten


MODEL_NAME = "conv_sentdex_l2reg=0_noise_drop-20-20-20-20*"
EPOCHS = 10
LEARNING_RATE = 0.001
PLOT = (true, 1)
HISTORY_TRAIN = (true, 1)
HISTORY_TEST = (true, 1)
USE_CUDA = false
# LOSS_ACCURACY_PORTION = 1.0
# LEARNING_RATE = 0.001

BATCH_SIZE = 256

MODEL() = Chain(
    Conv((5,1), 60 => 64, pad = SamePad(), relu),
    # Dropout(0.2),
    Conv((5,1), 64 => 128, pad = SamePad(), relu),
    # Dropout(0.2),
    Conv((5,1), 128 => 256, pad = SamePad(), relu),
    # Dropout(0.2),
    Conv((5,1), 256 => 512, pad = SamePad(), relu),
    # Dropout(0.2),
    Conv((16,1), 512 => 3),
    Flux.flatten
)
end