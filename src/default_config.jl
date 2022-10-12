# |--------------------------------------------------------------------------|
# | ARGUMENTS DATA                                                           |
# |--------------------------------------------------------------------------|

BATCH_SIZE = 256 # Size of mini-batches
NUM_CHANNELS = 16 # Number of EEG channels
MAX_FREQUENCY = 60 # Range of frequency produced by FFT (eg. here 1-60 Hz)

path_prefix = "../model_data/"
TRAIN_DATA = [
    (path_prefix * "directions/data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "directions/data/none/", [0.0, 1.0, 0.0]),
    (path_prefix * "directions/data/right/", [0.0, 0.0, 1.0])
] # (folder with files, desired outputs for each case)
TEST_DATA = [
    (path_prefix * "directions/validation_data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "directions/validation_data/none/", [0.0, 1.0, 0.0]),
    (path_prefix * "directions/validation_data/right/", [0.0, 0.0, 1.0])
] # (folder with files, desired outputs for each case)

SHUFFLE = true

# |----------------------------------------|
# | ARGUMENTS TRAINING                     |
# |----------------------------------------|

EPOCHS = 250
USE_CUDA = true
OPTIMIZER = ADAM
LEARNING_RATE = 0.00001
LOSS = Flux.logitcrossentropy
# Define model structure
MODEL() = Chain(
    Conv((3, 1), 60 => 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
    # softmax
)

LOAD_PATH = ""
SAVE_PATH = ""
MODEL_NAME = "*" # all stars get replaced by current date + time

# |----------------------------------------|
# | ARGUMENTS HISTORY                      |
# |----------------------------------------|

PLOT = (true, 5)

LOSS_ACCURACY_PORTION = 1.0 # 0.1: 10% of batches gets randomly selected to test loss and accuracy; 1.0: using all data

HISTORY_TRAIN = (true, 5)
HISTORY_TEST = (true, 5)

function gaussian(x)
    # Function for gaussian random noise, from https://fluxml.ai/tutorials/2021/02/07/convnet.html
    return x .+ device(0.25f0 * randn(eltype(x), size(x)))
end

NOISE = false
NOISE_FUNCTION = gaussian