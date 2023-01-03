export MCP3208, Ganglion
using BaremetalPi

struct FFTDataDescriptor <: DataDescriptor
    num_channels::Int
    freqs::Array{Int,1}
    sample_width::Int
end

struct RawDataDescriptor <: DataDescriptor
    num_channels::Int
    sample_width::Int
end


abstract type EEGBoard end
#===================#
#       DEVICES     #
#===================#
#------MCP3208------#
mutable struct MCP3208 <: EEGBoard
    spi_id::Int
    num_channels::Int
    online::Bool
    const data_descriptor::DataDescriptor
    sample
end

"""
    MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000, 
    online=true)

Initialise a MCP3208-based self-built EEG. `path` refers to the SPI 
path of the device and `num_channels` to the number of connected 
electrodes.

With `online` set to `true`, the EEG will be "simulated": `get_voltage(...)` 
will return random values.

TODO: max_speed_hz
"""
function MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000,
    online=true)
    if online
        id = init_spi(path, max_speed_hz=max_speed_hz)
    else
        id = 1
    end
    return MCP3208(id, num_channels, online, RawDataDescriptor(num_channels,
            num_channels), Array{Float64,1}(undef, num_channels))
end

function get_voltage(board::MCP3208, channel::Int)
    if channel < 1 || channel > board.num_channels
        throw(ArgumentError("Given channel smaller than 0 or 
        bigger than num of available channels on given board."))
    end
    if board.online
        # TODO: actually implement...
        tx_buf = [0x01, 0x80, 0x00]
        rx_buf = zeros(UInt8, 3)
        ret = spi_transfer!(board.spi_id, tx_buf, rx_buf)
        return (ret * 5) / flo
    else
        return 0.0 #(randn() - 0.5) * 10
    end
end
#------Ganglion------#
mutable struct GanglionGUI
    inlet#::StreamInlet{Float32}
    sample::Array{Float32,1}
    const data_descriptor::DataDescriptor
end

function GanglionGUI(num_channels::Int, max_freq::Int=125)
    streams = resolve_streams(timeout=2.0)
    inlet = StreamInlet(streams[1])
    sample = Array{Float32,1}(undef, max_freq * num_channels)
    # num_channel times to get one batch (all channels)
    for i = 1:num_channels
        # Index in sample where data of this channel starts
        channel_start = max_freq * (i - 1) + 1
        # ^^                                         ends
        channel_end = max_freq * i
        timestamp, sample[channel_start:channel_end] = pull_sample(board.inlet)
    end
    data_descriptor = FFTDataDescriptor(num_channels, max_freq,
        num_channels * max_freq)
    return GanglionGUI(inlet, sample, data_descriptor)
end

function get_sample!(board::GanglionGUI)
    for i = 1:board.data_descriptor.num_channels
        # Index in sample where data of this channel starts
        channel_start
        # Index in sample where data of this channel ends
        channel_end = max_freq * i
        pull_sample!(board.sample[channel_start:channel_end], board.inlet)
    end
end