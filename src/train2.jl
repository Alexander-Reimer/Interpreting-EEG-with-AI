module AI2

using Flux, CUDA

mutable struct Plot
    fig
    ax_loss
    ax_accuracy
    train_loss
    test_loss
    train_acc
    test_acc
end

mutable struct ModelParams
    network_builder::Function
    config_path::String
    η::Float32
    opt_type::DataType
    optimizer::Flux.Optimise.ADAM
    epochs_goal::Int
    cuda::Bool
    save_path::String
end

mutable struct EEGModel
    params::ModelParams
    net::Flux.Chain
    epochs_done::Int
    device::Function
end

"""
Create ModelParams instance using given configuration file.
"""
function create_params(config_path::String, default_config::String="default_config.jl")::ModelParams
    include(default_config)
    include(config_path)
    params = ModelParams(
        MODEL,
        config_path,
        LEARNING_RATE,
        OPTIMIZER,
        OPTIMIZER(LEARNING_RATE),
        EPOCHS,
        USE_CUDA,
        SAVE_PATH
    )
    return params
end

"""
Return CUDA device function (cpu or gpu).
"""
function get_device(params::ModelParams)::Function
    if params.cuda == true
        if CUDA.functional()
            return gpu
        else
            @warn "Despite USE_CUDA = true, CUDA is disabled as it isn't supported."
            return cpu
        end
    else
        return cpu
    end
end

function new_model(config_path::String, default_config::String="default_config.jl")
    params = create_params(config_path, default_config)
    net = params.network_builder()
    epochs_done = 0
    dev = get_device(params)

    # return EEGModel(MODEL(), config_path,)
end

params = create_params("config.jl")

end