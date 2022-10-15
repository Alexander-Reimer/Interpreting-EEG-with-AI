module EEG
#= 
using Pkg
Pkg.activate(".")
=#

using BaremetalPi
import PyPlot
Plt = PyPlot
Plt.pygui(true)

ONLINE = false

VIEW_WINDOW = 2000 # Show last 2000 ms in plot
PLOT_RES = 1
PLOT_FREQ = 1

if ONLINE
    init_spi("/dev/spidev0.0", max_speed_hz = 1000)
end

"""
Self-made EEG "board" using GPIO pins.
"""
mutable struct MCP3208
    spi_id::Int
    num_channels::Int
end

function MCP3208(path::String, num_channels::Int; max_speed_hz::Int = 1000)
    if ONLINE
        id = init_spi(path, max_speed_hz = max_speed_hz)
    else
        id = 1
    end
    return MCP3208(id, num_channels)
end

"""
Single EEG EEGChannel.
"""
mutable struct Channel
    times::Array{Float64}
    voltages::Array{Float64}
end

"""
EEG device.
"""
mutable struct Device
    board
    channels::Array{Channel}
    session_start::Float64
end

function Device(board::MCP3208)
    num_channels = board.num_channels
    channels = Array{Channel, 1}(undef, num_channels)
    for i = 1:num_channels
        channels[i] = Channel([], [])
    end
    return Device(board, channels, time())
end

"""
Get voltage from gpio pin.
"""
function get_voltage(board::MCP3208, channel::Int)
    if ONLINE
        tx_buf = [0x01, 0x80, 0x00]
        rx_buf = zeros(UInt8, 3)
        ret = spi_transfer!(1, tx_buf, rx_buf)
    else
        return randn() * 10
    end
end

"""
Get current voltages of device.
"""
function get_voltages(device::Device)
    channels = device.board.num_channels
    voltages = Array{Float64,1}(undef,channels)
    for c = 1:channels
        voltages[c] = get_voltage(device.board, c)
    end
    return voltages
end

function update_data!(device::Device)
    voltages = get_voltages(device)
    for index_c in eachindex(device.channels)
        channel = device.channels[index_c]
        if isempty(channel.voltages)
            device.session_start = time()
        end
        push!(channel.voltages, voltages[index_c])
        push!(channel.times, (time() - device.session_start) * 1000) # convert from s to ms
    end
end

function reset_data!(device::Device)
    for index_c in eachindex(device.channels)
        channel = device.channels[index_c]
        channel.voltages = Array{Float64, 1}(undef, 0)
        channel.times = Array{Float64, 1}(undef, 0)
    end
    device.session_start = time()
end

mutable struct ChannelLine
    line::Plt.PyCall.PyObject
end

function set_xdata(line::ChannelLine, xdata::AbstractArray)
    line.line.set_xdata(xdata)
end
function set_ydata(line::ChannelLine, ydata::AbstractArray)
    line.line.set_ydata(ydata)
end
function set_data(line::ChannelLine, xdata::AbstractArray, ydata::AbstractArray)
    line.line.set_data(xdata, ydata)
end

function add_xdata(line::ChannelLine, xdata::AbstractArray)
    # Get old data
    old_xdata = line.line.get_xdata()
    # Get type and assert new data matches type
    type = typeof(old_xdata)
    xdata::type

    l_old = length(old_xdata)
    total_l = l_old + length(xdata)
    # Initialise array with size = combined size of old data and new
    new_xdata = Array{type, 1}(undef, total_l)
    # Populate with old data
    for (i, val) in enumerate(old_xdata)
        new_xdata[i] = val
    end
    # Pupulate with new data
    for (i, val) in enumerate(xdata)
        new_xdata[i+l_old] = val
    end
    # Update data
    line.line.set_xdata(new_xdata)
end

function reset_data(line::ChannelLine)
    line.line.set_xdata([])
end
mutable struct Plot
    fig::Plt.Figure
    axes::Plt.PyCall.PyObject
    channel_lines::Array{ChannelLine}
    device::Device
    shown_channels::Array{Int}
    intervals::Int
end

function Plot(device::Device; interval=0, shown_channels=:all, title::String="raw EEG data")+
    num_channels = length(device.channels)
    if shown_channels == :all
        shown_channels = [i for i = 1:num_channels]
    end
    fig = Plt.figure(title)
    Plt.clf()
    axes = fig.subplots()

    axes.set_xlabel("Time in ms")
    axes.set_ylabel("Voltage in μV")
    axes.autoscale(true)
    
    channel_lines = Array{ChannelLine, 1}(undef, num_channels)
    for i = 1:num_channels
        line = Plt.plot([], [], label="Channel $i")[1]
        channel_lines[i] = ChannelLine(line)
    end
    axes.legend()
    
    return Plot(fig, axes, channel_lines, device, shown_channels, interval)
end

function compress_data(data, skip)
    return [data[i] for i = 1:skip:length(data)]
end

function update_plot!(plot::Plot)
    for c in eachindex(plot.channel_lines)
        line = plot.channel_lines[c]
        xdata, ydata = plot.device.channels[c].times, plot.device.channels[c].voltages
        xdata, ydata = compress_data(xdata, PLOT_RES), compress_data(ydata, PLOT_RES)
        set_data(line, xdata, ydata)
        # Plt.show()
    end
    latest_time = plot.device.channels[end].times[end]
    plot.axes.autoscale_view(scalex=false)
    plot.axes.set_xlim(latest_time-VIEW_WINDOW, latest_time)
    plot.axes.relim()
    Plt.draw()
end

device = Device(MCP3208("/dev/spidev0.0", 8))
plot = Plot(device)
reset_data!(device)

function main()
    for i = 1:10^9
        for i = 1:PLOT_FREQ
            update_data!(device)
        end
        update_plot!(plot)
        sleep(0.001)
        sleep(0)
    end
end

end # Module