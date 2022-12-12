module EEG
export Device
export Experiment, gather_data!
include("EEG_Devices.jl")
"""
Abstract class containing data processors.

Defined data processors:
`StandardProcessor`: The standard processors with preset arguments and functions, for details see [`StandardProcessor`](@ref).
"""
abstract type DataProcessor end
using BrainFlow, DataFrames, Dates
import PyPlot
Plt = PyPlot
Plt.pygui(true)

"""
Single Channel with voltages in Volt and the times the voltages were recorded at in seconds.
"""
mutable struct Channel
    times::Array{Float64}
    voltages::Array{Float64}
end

"""
Device for gathering EEG data. Create it using
    
Device(board::EEGBoard)

... TODO
"""
mutable struct Device
    board::EEGBoard
    channels::Array{Channel}
    session_start::Float64
end

function Device(board::EEGBoard)
    channels = Array{Channel,1}(undef, board.num_channels)
    for i = 1:board.num_channels
        channels[i] = Channel([], [])
    end
    return Device(board, channels, time())
end

function update_data!(device::Device)
    for channel_num in eachindex(device.channels)
        channel = device.channels[channel_num]
        push!(channel.voltages, get_voltage(device.board, channel_num))
        push!(channel.times, (time() - device.session_start) * 1000) # convert from s to ms
    end
    return 
end

"""
Standard configuration for processing EEG data. It uses a preset of functions and options and may not work for you.
Created using [`Standard`](@ref).

    Standard()
"""
struct StandardProcessor <: DataProcessor
    output_type::Dict{Symbol,Any}
    fft_maxfreq::Integer
end
"""
    Standard()::StandardProcessor

Create standard configuration for processing EEG data. See [`StandardProcessor`](@ref) for more details.
"""
function Standard()
    output_type = Dict()
    output_type[:type] = :SpectralDensity # requires output dims (num_channels, max_freq)
    processor = StandardProcessor(output_type, 60)
    return processor
end

"""
TODO: Processing

    1. Artifacts Removal
    2. Data Filtering
    3. Feature Extraction (FFT)
Also see [this](https://www.sciencedirect.com/science/article/pii/S1877705812022114/pdf?md5=3c7a2bdf5717d518cf46c4ef5d145d33&pid=1-s2.0-S1877705812022114-main.pdf), page 2529.
"""
function process(device::Device, processor::StandardProcessor)
    result = Array{Float64,2}(undef, device.board.num_channels, processor.fft_maxfreq)
    # Temporarily skip all processing
    # TODO
    for i in eachindex(result)
        result[i] = rand()
    end
    return result
end

struct Experiment
    device::Device
    raw_data::DataFrame
    tags::Array{String,1}
    extra_info::Dict{Symbol,Any}
    path::String
    # cases::Array{Symbol,1}
end

function datetime(seconds_since_epoch)::DateTime
    return Dates.unix2datetime(seconds_since_epoch)
end

"""
`name`: Name of the experiment (e.g. "BlinkDetection").

`tags`: Tags which will be applied to all data gathered with this Experiment.

`path`: To top-level of data directory (e.g. "data/"). If empty, files can't be saved.

TODO: `load_previous` not implemented yet (maybe in another function?)
"""
function Experiment(device::Device, name::String; tags::Array=[], extra_info::Dict=Dict(), path::String="data/", load_previous::Bool=false)
    column_names = ["time", "tags", "extraInfo"]
    num_channels = length(device.channels)
    for channel = 1:num_channels
        push!(column_names, string(channel))
    end
    types = fill(Float64[], num_channels)
    #        Time       tags      extra_info    voltages  
    types = [Float64[], Array{Array{String}, 1}(), Dict[],       types...]
    raw_data = DataFrame(types, column_names)
    file_path = joinpath(path, name, "raw_data.csv")
    experiment = Experiment(device, raw_data, string.(tags), extra_info, file_path)
    return experiment
end

function append_raw_data(data_io, data::Array{Number,2})

end

function get_voltages(board::EEGBoard)
    voltages = Array{Float64, 1}(undef, board.num_channels)
    for channel in eachindex(voltages)
        voltages[channel] = get_voltage(board, channel)
    end
    return time(), voltages
end

"""
    gather_data(experiment::Experiment, time::Number; tags::Array{String, 1} = [])

Gather raw EEG data. `time` is in seconds.

TODO: create test
"""
function gather_data!(experiment::Experiment, runtime::Number; tags::Array=[], extra_info::Dict=Dict())
    all_tags = vcat(experiment.tags, string.(tags))
    new_row = Array{Any,1}(undef, 3 + experiment.device.board.num_channels)
    new_row[2] = all_tags
    new_row[3] = merge(experiment.extra_info, extra_info)
    start_time = time()
    while (time() - start_time) < runtime
        new_row[1] = time()
        for channel = 1:experiment.device.board.num_channels
            new_row[3+channel] = get_voltage(experiment.device.board, channel)
        end
        push!(experiment.raw_data, new_row)
    end
end

function save_data(experiment)
    
end

function get_sample()

end

struct DataFilter
    include_tags::Array{String,1}
    exclude_tags::Array{String,1}
    extra_info_filter::Function
end

struct DataHandler
    data_processor::DataProcessor
    data_io#::DataIO
    cases::Union{Array{Symbol,1},Nothing}
    name::Union{String,Nothing}
    max_freq::Union{Int,Nothing}
end

"""
    DataHandler(data_processor::DataProcessor, data_io::DataIO; cases=nothing, name=nothing, max_freq=nothing)

Create a DataHandler instance. `cases`, `name` and `max_freq` are automatically determined by
data saved at path if they are `nothing`.

Example:

    data_io = DataIO("data/test", states)
    data_handler = DataHandler(data_io, StandardFFT())
"""
function DataHandler(data_processor::DataProcessor, data_io; cases=nothing, name=nothing, max_freq=nothing)

end
end