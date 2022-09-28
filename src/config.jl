# LOAD_PATH = "saved_models/amodel3.bson"
SAVE_PATH = "saved_models/amodel4.bson"
EPOCHS = 20
USE_CUDA = false
PLOT = (true, 2)
HISTORY_TRAIN = (true, 2)
HISTORY_TEST = (true, 2)
LOSS_ACCURACY_PORTION = 1.0
LEARNING_RATE = 0.001

BATCH_SIZE = 256

#= MODEL() = Chain(
    Conv((5, 1), 16 => 64, relu),
    Conv((5, 1), 64 => 128, relu),
    Conv((5, 1), 128 => 256, relu),
    Conv((5, 1), 256 => 512, relu),
    Conv((16, 1), 512 => 3, relu),
    Flux.flatten,
    Dense(87, 3, tanh),
    softmax
) =#

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

# MODEL() = Chain(
#     Dropout(0.05),
#     Conv((3, 1), 16 => 32, relu),
#     MaxPool((2, 1)),
#     Flux.flatten,
#     Dense(928, 64, tanh),
#     Dense(64, 3, tanh)
# )

MODEL() = Chain(
    Conv((7, 1), 16=> 64, relu),
    Dropout(0.5),
    MaxPool((2, 1)),
    Conv((2, 1), 64=> 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dense(208, 32, tanh),
    Dense(32, 3)
)
w_layers = [2, 5]
#= MODEL() = Chain(
    Conv((5,1), 16 => 64, pad = SamePad(), relu),
    Conv((5,1), 64 => 128, pad = SamePad(), relu),
    Conv((5,1), 128 => 256, pad = SamePad(), relu),
    Conv((5,1), 256 => 512, pad = SamePad(), relu),
    Conv((16,1), 512 => 3),
    Flux.flatten,
    softmax
) =#