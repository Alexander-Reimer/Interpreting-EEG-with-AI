module EEG
export MCP3208, Device, Ganglion, gather_data
abstract type EEGBoard end
"""
Abstract class containing data processors.
`StandardProcessor`: The standard processors with preset arguments and functions.
"""
abstract type DataProcessor end
using BaremetalPi, BrainFlow, DataFrames
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
end

"""
    Standard()::StandardProcessor
For more details, see

    Standard()
"""
struct StandardProcessor <: DataProcessor
    output_type::Dict{Symbol,Any}
    fft_maxfreq::Integer
end
"""
    Standard()::StandardProcessor

Create standard configuration for processing EEG data. It uses a preset of functions and options and may not work for you.
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

struct DataIO
    path::String
    cases::Array{Symbol,1}
end

"""
`path`: To top-level of data directory (e.g. "data/")
`name`: Name for the data (e.g. "BlinkDetection")
"""
function DataIO(path::String, name::String)
    
end

function append_raw_data(data_io::DataIO, data::Array{Number, 2})
    
end

# TODO: create test
function gather_data(device::Device)#, data_io, case, time::Number; tags::Array{Symbol, 1}=[])
    start_time = time()
    column_names = Array{String, 1}(undef, 0)
    num_channels = length(device.channels)
    for channel = 1:num_channels
        push!(column_names, string(channel))
    end
    push!(column_names, "tags")
    types = [Float64, Float64]
    types = [types..., String[]]
    raw_data = DataFrame(types, column_names)
    return raw_data
    # while time() - start_time < time

    # end
end

struct DataHandler
    data_processor::DataProcessor
    data_io::DataIO
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
function DataHandler(data_processor::DataProcessor, data_io::DataIO; cases=nothing, name=nothing, max_freq=nothing)

end

#===================#
#       DEVICES     #
#===================#
#------MCP3208------#
mutable struct MCP3208 <: EEGBoard
    spi_id::Int
    num_channels::Int
    online::Bool
end
"""
    MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000, online=true)

Initialise a MCP3208-based self-built EEG. `path` refers to the SPI path of the device 
and `num_channels` to the number of connected electrodes.

With `online` set to `true`, the EEG will be "simulated": `get_voltage(...)` will return 
random values.

TODO: max_speed_hz
"""
function MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000, online=true)
    if online
        id = init_spi(path, max_speed_hz=max_speed_hz)
    else
        id = 1
    end
    return MCP3208(id, num_channels, online)
end
function get_voltage(board::MCP3208, channel::Int)
    if channel < 1 || channel > board.num_channels
        throw(ArgumentError("Given channel smaller than 0 or bigger than num of available channels on given board."))
    end
    if board.online
        # TODO: actually implement...
        tx_buf = [0x01, 0x80, 0x00]
        rx_buf = zeros(UInt8, 3)
        ret = spi_transfer!(board.spi_id, tx_buf, rx_buf)
        return (ret * 5) / flo
    else
        return (randn() - 0.5) * 10
    end
end
end