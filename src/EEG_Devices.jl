export MCP3208, Ganglion
using BaremetalPi
abstract type EEGBoard end
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
        return 0.0 #(randn() - 0.5) * 10
    end
end