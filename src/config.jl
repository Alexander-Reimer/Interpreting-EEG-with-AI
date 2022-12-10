using Flux
include("default_config.jl") # Load default configuration, to be overwritten
config = init_config()
config.EPOCHS = 10
config.MODEL_NAME = "allConv_4c_1_*"
config.BATCH_SIZE = 128
config.CHANNELS_USED = [12, 10, 9, 11]
# config.CHANNELS_USED = [12]
config.MODEL = Chain(
        # @autosize begin
        Conv((5,1), 60 => 64, pad = SamePad(), relu),
        Conv((5,1), 64 => 64, pad = SamePad(), relu),
        Conv((5,1), 64 => 128, pad = SamePad(), relu),
        Conv((5,1), 128 => 256, pad = SamePad(), relu),
        Conv((5,1), 256 => 512, pad = SamePad(), relu),
        Conv((length(config.CHANNELS_USED),1), 512 => 3),
        Flux.flatten,
    )
config.LEARNING_RATE = 0.00005
config.PRUNE_FREQUENCY = 10
path_prefix = "holy_"
config.TEST_DATA = [
    (path_prefix * "data/left/", [1.0, 0.0, 0.0]),
    (path_prefix * "data/none/", [0.0, 1.0, 0.0]),
    (path_prefix * "data/right/", [0.0, 0.0, 1.0])
    ]
config.USE_CUDA = true
#= TEST_DATA = [
        (path_prefix * "directions/validation_data/left/", [1.0, 0.0, 0.0]),
        (path_prefix * "directions/validation_data/none/", [0.0, 1.0, 0.0]),
        (path_prefix * "directions/validation_data/right/", [0.0, 0.0, 1.0])
    ] =#


#=
mutable struct Config_struct
    # LOAD_PATH = "saved_models/amodel5.bson"
    SAVE_PATH :: String
    MODEL_NAME :: String 
    EPOCHS :: Int
    LEARNING_RATE :: Float64
    # USE_CUDA = true
    PLOT :: Tuple
    HISTORY_TRAIN :: Tuple
    HISTORY_TEST :: Tuple
    # LOSS_ACCURACY_PORTION = 1.0
    # LEARNING_RATE = 0.001

    BATCH_SIZE :: Int


    MODEL :: Chain
end
=#
#=
function init_config()
    return Config_struct("", "conv_sentdex_*", 25, 0.001, (true, 1), (true, 1), (true, 1), 512, Chain(
        Conv((5,1), 60 => 64, pad = SamePad(), relu),
        Conv((5,1), 64 => 128, pad = SamePad(), relu),
        Conv((5,1), 128 => 256, pad = SamePad(), relu),
        Conv((5,1), 256 => 512, pad = SamePad(), relu),
        Conv((16,1), 512 => 3),
        Flux.flatten#,
        # softmax
    ))
end
=#

#= config.MODEL = Chain(
    Conv((3,1), 60 => 64, relu),
    Conv((2,1), 64 => 128, relu),
    Conv((2,1), 128 => 128, relu),
    Conv((2,1), 128 => 64, relu),
    MaxPool((2,1)),
    flatten,
    Dense(320, 512),
    Dense(512, 256),
    Dense(256, 128),
    Dense(128, 3)
    ) =#

# MODEL() = Chain(
#     Conv((5, 1), 16 => 64, relu),
#     Conv((5, 1), 64 => 128, relu),
#     Conv((5, 1), 128 => 256, relu),
#     Conv((5, 1), 256 => 32, relu),
#     Conv((3, 1), 32 => 3, relu),
#     Flux.flatten,
#     Dense(3, 50),
#     Dense(50, 3, tanh),
#     softmax
# )
# w_layers = [1,2,3,4,7]

# MODEL() = Chain(
#     Conv((5, 1), 16 => 64, relu),
#     Conv((5, 1), 64 => 128, relu),
#     Conv((5, 1), 128 => 256, relu),
#     Conv((5, 1), 256 => 512, relu),
#     Conv((16, 1), 512 => 3, relu),
#     Flux.flatten,
#     Dense(87, 50),
#     Dense(50, 3, tanh),
#     softmax
# )
# w_layers = [1,2,3,4,7]

#= MODEL() = Chain(
    Conv((5, 1), 16 => 64, relu),
    Conv((5, 1), 64 => 128, relu),
    Conv((5, 1), 128 => 256, relu),
    Conv((16, 1), 256 => 3, relu),
    Flux.flatten,
    Dense(99, 50),
    Dense(50, 3, tanh),
    softmax
)
w_layers = [1,2,3,7] =#

#= MODEL() = Chain(
    Dropout(0.05),
    Conv((3, 1), 16 => 32, relu),
    Conv((5, 1), 32 => 64, relu),
    Conv((7, 1), 64 => 64, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dense(1536, 64, tanh),
    Dense(64, 3, tanh)
) =#

#= MODEL() = Chain(
    Dropout(0.05),
    Conv((3, 1), 16 => 32, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dense(928, 64, tanh),
    Dense(64, 3, tanh)
)
w_layers = [] =#

#= MODEL() = Chain(
    Conv((7, 1), 16=> 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dense(16, 16, tanh),
    Dense(16, 3)
)
w_layers = [1, 7] =#