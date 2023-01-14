using Flux
include("default_config.jl")

config = init_config()
config.EPOCHS = 50
config.BATCH_SIZE = 128
# config.CHANNELS_USED = [12, 10, 9, 11]
config.CHANNELS_USED = [i for i = 1:16]
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
    Conv((2, 1), 64 => 32, pad = SamePad(), relu),
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
    Dense(16, 3),
)

config.MODEL = modelOmegaH

config.LEARNING_RATE = 0.00001
config.MODEL_NAME = "OmegaH_16c_sentdex_e" * string(config.LEARNING_RATE) * "_*"
