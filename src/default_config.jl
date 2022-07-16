# |--------------------------------------------------------------------------|
# | ARGUMENTS DATA                                                           |
# |--------------------------------------------------------------------------|

BATCH_SIZE = 32
NUM_CHANNELS = 16 # Number of EEG channels
MAX_FREQUENCY = 60 # Range of frequency produced by FFT (eg. here 1-60 Hz)
NUM_SAMPLES_TRAIN = 284375 # Total number of training samples
NUM_SAMPLES_TEST = 35250 # Total number of testing samples

path_prefix = "../model_data/"
TRAIN_DATA = [
    (path_prefix * "directions/data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "directions/data/right/", [0.0, 1.0, 0.0]),
    (path_prefix * "directions/data/left/", [0.0, 0.0, 1.0])
] # (folder with files, desired outputs for each case)
TEST_DATA = [
    (path_prefix * "directions/validation_data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "directions/validation_data/right/", [0.0, 1.0, 0.0]),
    (path_prefix * "directions/validation_data/left/", [0.0, 0.0, 1.0])
] # (folder with files, desired outputs for each case)


# |----------------------------------------|
# | ARGUMENTS TRAINING                     |
# |----------------------------------------|

OPTIMIZER = ADAM
LEARNING_RATE = 0.001
LOSS = crossentropy
# Define model structure
MODEL() = Chain(
    Conv((3, 1), 16 => 64, relu),
    Conv((2, 1), 64 => 128, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 128 => 64, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 64, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dense(384, 256, tanh),
    Dense(256, 128, tanh),
    Dense(128, 3),
    softmax
)