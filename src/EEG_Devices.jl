"""
Supertpye of all EEG boards.

When creating your own EEG board, you should make it a subtype of `AbstractEEGBoard` with

```julia
struct MyBoard <: AbstractEEGBoard
    ...
end
```
"""
abstract type EEGBoard end

"""
    get_sample!(board::EEGBoard)

Updates board.sample to new data from board.
"""
function get_sample!(board::EEGBoard)
    for channel in 1:(board.data_descriptor.num_channels)
        board.sample[channel] = get_voltage(board, channel)
    end
end

"""
Abstract type containing data descriptors.

Data is processed and trained on differently depending on the data descriptor.
If you store or use your data in a format which doesn't have a data descriptor
yet, you will need to define your own and overload the following functions:

`create_data(name::String, data_desc::MyDataDescriptor)`

`get_inputshape(data_desc::MyDataDescriptor)`

`inputarray(data_descriptor::MyDataDescriptor, num_samples::Int)`

`row2inputs(data_descriptor::MyDataDescriptor, row::Vector)`
"""
abstract type AbstractDataDescriptor end

struct FFTDataDescriptor <: AbstractDataDescriptor
    num_channels::Int
    max_freq::Int
    sample_width::Int
end
copy(x::FFTDataDescriptor) = FFTDataDescriptor(x.num_channels, x.max_freq, x.sample_width)

"""
    FFTDataDescriptor(num_channels::Int, max_freq::Int)

Data descriptor for FFT data in the format of a 2D-Array per sample, with the first
dimension corresponding to the frequncies from 1 to max_freq and the seconds dimension
corresponding to the channels from 1 to num_channels.
"""
function FFTDataDescriptor(num_channels::Int, max_freq::Int)
    return FFTDataDescriptor(num_channels, max_freq, num_channels * max_freq)
end

struct RawDataDescriptor <: AbstractDataDescriptor
    num_channels::Int
    sample_width::Int
end
copy(x::RawDataDescriptor) = RawDataDescriptor(x.num_channels, x.sample_width)

"""
    RawDataDescriptor(num_channels::Int)

Data descriptor for raw data in the format of a 1D-Array per sample, with each channels
voltage in a row.

`num_channels::Int`: Number of channels the data has (= width of the row).
"""
RawDataDescriptor(num_channels::Int) = RawDataDescriptor(num_channels, num_channels)

#===================#
#       DEVICES     #
#===================#
#------MCP3208------#
mutable struct MCP3208 <: EEGBoard
    spi_id::Int
    num_channels::Int
    online::Bool
    const data_descriptor::AbstractDataDescriptor
    sample
end

"""
    MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000,
    online=true)

Initialise a MCP3208-based self-built EEG.

`path::String`: SPI path of the device

`num_channels::Int`: Number of connected electrodes.

`max_speed_hz::Int`: 1000 by default. Tells the Raspberry Pi (or similar device) how often
per second the output should be read.
# TODO: explain (side) effects

`online::Bool`: `true` by default; if set to `false`, `get_voltage(...)` will just return 0.
This is for testing all associated functions and devices using this board without having it
connected (e.g. in tests).
"""
function MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000, online=true)
    if online
        id = init_spi(path; max_speed_hz=max_speed_hz)
    else
        id = 1
    end
    return MCP3208(
        id,
        num_channels,
        online,
        RawDataDescriptor(num_channels),
        Array{Float64,1}(undef, num_channels),
    )
end

"""
    get_voltage(board::MCP3208, channel::Int)

Read digital output of the ADC on channel `channel`.

If MCP3208 was given `online=false` at creation, return 0 instead.
"""
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
mutable struct GanglionGUI <: EEGBoard
    inlet#::StreamInlet{Float32}
    sample::Array{Float32,1}
    const data_descriptor::AbstractDataDescriptor
end

function GanglionGUI(num_channels::Int)
    streams = resolve_streams(; timeout=2.0)
    if length(streams) == 0
        throw(ErrorException("No stream found!
        Make sure you have it enabled in the Ganglion GUI"))
    end
    inlet = StreamInlet(streams[1])
    max_freq::Int = channel_count(inlet.info)
    sample = Array{Float32,1}(undef, max_freq * num_channels)
    # num_channel times to get one batch (all channels)
    for i in 1:num_channels
        # Index in sample where data of this channel starts
        channel_start = max_freq * (i - 1) + 1
        # ^^                                         ends
        channel_end = max_freq * i
        timestamp, sample[channel_start:channel_end] = pull_sample(inlet; timeout=3)
    end
    data_descriptor = FFTDataDescriptor(num_channels, max_freq)
    return GanglionGUI(inlet, sample, data_descriptor)
end

function get_sample!(board::GanglionGUI)
    max_freq = board.data_descriptor.max_freq
    for i in 1:(board.data_descriptor.num_channels)
        # Index in sample where data of this channel starts
        channel_start = max_freq * (i - 1) + 1
        # Index in sample where data of this channel ends
        channel_end = max_freq * i
        pull_sample!(board.sample[channel_start:channel_end], board.inlet)
    end
end
