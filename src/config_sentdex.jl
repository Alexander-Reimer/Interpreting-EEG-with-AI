using Flux
include("default_config.jl")

config = init_config()
config.EPOCHS = 50
config.BATCH_SIZE = 128
# Model A bzw. allConv(_4c)_1)
#= config.MODEL = Chain(
    # @autosize begin
        Conv((5,1), 60 => 64, pad = SamePad(), relu),
        Conv((5,1), 64 => 64, pad = SamePad(), relu),
        Conv((5,1), 64 => 128, pad = SamePad(), relu),
        Conv((5,1), 128 => 256, pad = SamePad(), relu),
        Conv((5,1), 256 => 512, pad = SamePad(), relu),
        Conv((length(config.CHANNELS_USED),1), 512 => 3),
        Flux.flatten,
        ) =#

modelOmegaA = Chain( # Model Omega A (+1 Dense(16,16))
    Conv((2, 1), 60 => 64, relu, pad = SamePad()),
    Dropout(0.2),
    # MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, relu, pad = SamePad()),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu, pad = SamePad()),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 16, tanh),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaB = Chain( # Model Omega B (zu A: +1 Dense(16,16))
    Conv((3, 1), 60 => 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 16, tanh),
    Dense(16, 16, tanh),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaC = Chain( # Model Omega C
    Conv((3, 1), 60 => 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 16, tanh),
    Dense(16, 32, tanh),
    Dense(32, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

config.MODEL = Chain( # Model Omega D
    Conv((3, 1), 60 => 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 16, tanh),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

config.MODEL = Chain( # Model Omega E
    Conv((3, 1), 60 => 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, pad=SamePad(), relu),
    Conv((2, 1), 32 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 16, tanh),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaF = Chain( # Model Omega F (reduced from Omeg A)
    Conv((3, 1), 60 => 32, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    # Conv((2, 1), 32 => 32, pad=SamePad(), relu),
    Conv((2, 1), 32 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 16, tanh),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaG = Chain( # Model Omega G (reduced from Omeg F)
    Conv((3, 1), 60 => 64, relu),
    Dropout(0.2),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((2, 1), 32 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 24, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(24, 100, tanh),
    Dropout(0.15),
    Dense(100, 64, tanh),
    Dropout(0.15),
    Dense(64, 64, tanh),
    Dropout(0.15),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaH = Chain( # Model Omega G (reduced from Omeg F)
    Conv((3, 1), 60 => 64, relu),
    # Dropout(0.15),
    MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((2, 1), 32 => 32, relu),
    MaxPool((2, 1)),
    Conv((2, 1), 32 => 24, relu),
    MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.1),
    Dense(24, 100, tanh),
    Dense(100, 64, tanh),
    Dense(64, 64, tanh),
    Dropout(0.1),
    Dense(64, 16, tanh),
    Dense(16, 3, tanh),
)

modelOmegaI = Chain(
    Conv((3, 1), 60 => 64, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Conv((2, 1), 64 => 32, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Conv((2, 1), 32 => 16, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaJ = Chain(
    Conv((3, 1), 60 => 64, relu),
    MaxPool((2, 1)),
    Dropout(0.3),
    Conv((2, 1), 64 => 16, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Flux.flatten,
    Dropout(0.2),
    Dense(48, 64, tanh),
    Dropout(0.4),
    Dense(64, 16, tanh),
    Dense(16, 3),
)

modelOmegaK = Chain(
    Conv((3, 1), 60 => 64, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Conv((2, 1), 64 => 32, relu),
    Conv((2, 1), 32 => 32, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Conv((2, 1), 32 => 16, relu),
    Dropout(0.4),
    Conv((2, 1), 16 => 16, relu),
    MaxPool((2, 1)),
    Dropout(0.2),
    Flux.flatten,
    Dropout(0.2),
    Dense(16, 3),
)

modelOmegaL = Chain( # Model Omega L (for Gang data)
    Conv((2, 1), 60 => 64, relu, pad=SamePad()),
    Dropout(0.2),
    # MaxPool((2, 1)),
    Conv((2, 1), 64 => 32, relu, pad=SamePad()),
    # MaxPool((2, 1)),
    Conv((2, 1), 32 => 32, relu, pad=SamePad()),
    # MaxPool((2, 1)),
    Flux.flatten,
    Dropout(0.2),
    Dense(128, 64, tanh),
    Dense(64, 16, tanh),
    Dense(16, 32, tanh),
    # Dense(64, 16, tanh),
    Dense(32, 3),
)

modelOmegaM = Chain( # Model Omega L (for Gang data)
    # Dropout(0.05),
    Flux.flatten,
    Dense(240, 480, tanh),
    Dropout(0.2),
    Dense(480, 32, tanh),
    Dropout(0.2),
    Dense(32, 64, tanh),
    Dropout(0.1),
    # Dense(64, 16, tanh),
    Dense(64, 3),
)

modelOmegaN = Chain(
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
)

modelAlpha = Chain(
    # @autosize begin
    Conv((5, 1), 60 => 64, pad=SamePad(), relu),
    Conv((5, 1), 64 => 64, pad=SamePad(), relu),
    Conv((5, 1), 64 => 128, pad=SamePad(), relu),
    Conv((5, 1), 128 => 256, pad=SamePad(), relu),
    Conv((5, 1), 256 => 512, pad=SamePad(), relu),
    Conv((length(config.CHANNELS_USED), 1), 512 => 3),
    Flux.flatten,
)

config.CHANNELS_USED = [12, 10, 9, 11]
config.CHANNELS_USED = [i for i = 1:16]

modelAlphaB = Chain(
    # @autosize begin
    Dropout(0.05),
    Conv((5, 1), 60 => 64, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((5, 1), 64 => 64, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((5, 1), 64 => 128, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((5, 1), 128 => 256, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((5, 1), 256 => 512, pad=SamePad(), relu),
    Dropout(0.1),
    Conv((length(config.CHANNELS_USED), 1), 512 => 3),
    Flux.flatten,
)

# path_prefix = "holy_"
# config.TEST_DATA = [
#     (path_prefix * "data/left/", [1.0, 0.0, 0.0]),
#     (path_prefix * "data/none/", [0.0, 1.0, 0.0]),
#     (path_prefix * "data/right/", [0.0, 0.0, 1.0])
# ]

config.MODEL = modelAlphaB

config.LEARNING_RATE = 0.00001
config.MODEL_NAME = "AlphaB_16c_gangTest_e" * string(config.LEARNING_RATE) * "_*"
