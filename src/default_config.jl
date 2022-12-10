mutable struct ConfigStruct
    # |--------------------------------------------------------------------------|
    # | ARGUMENTS DATA                                                           |
    # |--------------------------------------------------------------------------|
    BATCH_SIZE :: Int # Size of mini-batches
    NUM_CHANNELS :: Int # Number of EEG channels
    CHANNELS_USED :: Array
    MAX_FREQUENCY :: Int # Range of frequency produced by FFT (eg. here 1-60 Hz)
    path_prefix :: String
    TRAIN_DATA :: Array # (folder with files, desired outputs for each case)
    TEST_DATA :: Array # (folder with files, desired outputs for each case)
    SHUFFLE :: Bool
    CLIP :: Int
    # |----------------------------------------|
    # | ARGUMENTS TRAINING                     |
    # |----------------------------------------|
    EPOCHS :: Int
    USE_CUDA :: Bool
    OPTIMIZER :: DataType
    LEARNING_RATE :: Float64
    LOSS :: Function
    # Define model structure
    MODEL :: Chain 
    LOAD_PATH :: String
    MODEL_NAME :: String # all stars get replaced by current date + time
    # |----------------------------------------|
    # | ARGUMENTS HISTORY                      |
    # |----------------------------------------|
    PLOT :: Tuple
    LOSS_ACCURACY_PORTION :: Float64  # 0.1: 10% of batches gets randomly selected to test loss and accuracy; 1.0: using all data
    HISTORY_TRAIN :: Tuple
    HISTORY_TEST :: Tuple
    NOISE_FUNCTION :: Function
    NOISE :: Bool
    PRUNE_GUARD :: Array
    PRUNE_FREQUENCY :: Int
    PRUNE_AMOUNT :: Int
end

function gaussian(x)
    return x .+ device(0.25f0 * randn(eltype(x), size(x)))
end

function init_config(; BATCH_SIZE = 256, NUM_CHANNELS = 16, CHANNELS_USED = [1,2,3,4], MAX_FREQUENCY = 60, path_prefix = "../model_data/", TRAIN_DATA = [
    (path_prefix * "directions/data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "directions/data/none/", [0.0, 1.0, 0.0]),
    (path_prefix * "directions/data/right/", [0.0, 0.0, 1.0])
    ], TEST_DATA = [
        (path_prefix * "directions/validation_data/left/", [1.0, 0.0, 0.0]),
        (path_prefix * "directions/validation_data/none/", [0.0, 1.0, 0.0]),
        (path_prefix * "directions/validation_data/right/", [0.0, 0.0, 1.0])
    ], SHUFFLE = true, CLIP = 20, EPOCHS = 1, USE_CUDA = true, OPTIMIZER = Flux.Optimise.ADAM, LEARNING_RATE = 0.00001, LOSS = Flux.logitcrossentropy, MODEL = Chain(
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
    ), LOAD_PATH = "", SAVE_PATH = "", MODEL_NAME = "*", PLOT = (true, 5), LOSS_ACCURACY_PORTION = 1.0, HISTORY_TRAIN = (true, 5), 
    HISTORY_TEST = (true, 5), NOISE_FUNCTION = gaussian, NOISE = false, PRUNE_GUARD = [], PRUNE_FREQUENCY = 5,
    PRUNE_AMOUNT = 1)

    return ConfigStruct(BATCH_SIZE, 
    NUM_CHANNELS,
    CHANNELS_USED,
    MAX_FREQUENCY,
    path_prefix, 
    TRAIN_DATA, 
    TEST_DATA, 
    SHUFFLE, 
    CLIP, 
    EPOCHS, 
    USE_CUDA, 
    OPTIMIZER, 
    LEARNING_RATE, 
    LOSS, 
    MODEL,
    LOAD_PATH,
    MODEL_NAME,
    PLOT,
    LOSS_ACCURACY_PORTION,
    HISTORY_TRAIN,
    HISTORY_TEST,
    NOISE_FUNCTION,
    NOISE,
    PRUNE_GUARD,
    PRUNE_FREQUENCY,
    PRUNE_AMOUNT)
end
