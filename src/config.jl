module Config
using Flux
include("default_config.jl") # Load default configuration, to be overwritten


MODEL_NAME = "conv_sentdex_l2reg=0_noise_drop-20-20-20-20*"
EPOCHS = 25
LEARNING_RATE = 0.001
PLOT = (true, 1)
HISTORY_TRAIN = (true, 1)
HISTORY_TEST = (true, 1)

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

#= 
MODEL() = Chain(
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
    Dense(208, 32, tanh),
    Dense(32, 3)
)
w_layers = [1, 7] =#
end