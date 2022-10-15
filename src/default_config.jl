mutable struct Config_struct
    # |--------------------------------------------------------------------------|
    # | ARGUMENTS DATA                                                           |
    # |--------------------------------------------------------------------------|

    BATCH_SIZE :: Int # Size of mini-batches
    NUM_CHANNELS :: Int # Number of EEG channels
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
    SAVE_PATH :: String
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
    #NOISE_FUNCTION :: Function 
end

function gaussian(x)
    return x .+ device(0.25f0 * randn(eltype(x), size(x)))
end

function init_config()
    path_prefix = "../model_data/"
    return Config_struct(256, 
    16,
    60,
    "../model_data/", [
    (path_prefix * "directions/data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "directions/data/none/", [0.0, 1.0, 0.0]),
    (path_prefix * "directions/data/right/", [0.0, 0.0, 1.0])
    ], 
    [
        (path_prefix * "directions/validation_data/left/", [1.0, 0.0, 0.0]),
        (path_prefix * "directions/validation_data/none/", [0.0, 1.0, 0.0]),
        (path_prefix * "directions/validation_data/right/", [0.0, 0.0, 1.0])
    ], 
    true, 
    20, 
    1, 
    true, 
    ADAM, 
    0.00001, 
    Flux.logitcrossentropy, 
    Chain(
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
    ), 
    "", 
    "", 
    "*", 
    (true, 5), 
    1.0, 
    (true, 5), 
    (true, 5), 
    gaussian, 
    false)
end