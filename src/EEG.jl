module EEG
export MCP3208, Device
abstract type EEGBoard end
"""
Abstract class containing data processors.
`StandardProcessor`: The standard processors with preset arguments and functions.
"""
abstract type DataProcessor end
using BaremetalPi
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

struct DataHandler
    data_processor::DataProcessor
    cases::Array{Symbol,1}
    path::String
    max_freq::Int
end

# TODO: create test
function gather_data(device::Device, data_handler::DataHandler, case::Symbol, time::Float64)
    start_time = time()
    while time() - start_time < time
        # TODO
    end
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