module EEG
export MCP3208, Device
abstract type EEGBoard end
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
    num_channels = board.num_channels
    channels = Array{Channel,1}(undef, num_channels)
    for i = 1:num_channels
        channels[i] = Channel([], [])
    end
    return Device(board, channels, time())
end

"""
Get current voltages of device.
"""
function get_voltages(device::Device)
    channels = device.board.num_channels
    voltages = Array{Float64,1}(undef, channels)
    for c = 1:channels
        voltages[c] = get_voltage(device.board, c)
    end
    return voltages
end

struct FFTProcessor <: DataProcessor
    
end
function process(voltages, processor::FFTProcessor)
    
end
function StandardFFT()
    return FFTProcessor()
end

struct DataHandler
    data_processor::DataProcessor
    cases::Array{Symbol, 1}
    path::String
    max_freq::Int
end

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