module EEG

# using Dates
using PyPlot

"""
Self-made EEG "board" using GPIO pins.
"""
mutable struct EEG3000
    gpios::Array{Int,1}
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

function Device(board::EEG3000)
    num_channels = length(board.gpios)
    channels = Array{Channel, 1}(undef, num_channels)
    for i = 1:num_channels
        channels[i] = Channel([], [])
    end
    return Device(board, channels, time())
end

"""
Get voltage from gpio pin.
"""
function get_voltage(gpio::Int)
    return randn() * 10
end

"""
Get current voltages of device.
"""
function get_voltages(device::EEG3000)
    channels = length(device.gpios)
    voltages = Array{Float64,1}(undef,channels)
    for c in eachindex(device.gpios)
        voltages[c] = get_voltage(device.gpios[c])
    end
    return voltages
end

function update_data!(device::Device)
    voltages = get_voltages(device.board)
    for index_c in eachindex(device.channels)
        channel = device.channels[index_c]
        if isempty(channel.voltages)
            device.session_start = time()
        end
        push!(channel.voltages, voltages[index_c])
        push!(channel.times, (time() - device.session_start) / 1000) # convert from s to ms
    end
end

mutable struct ChannelLine
    line
end
mutable struct Plot
    fig
    axes
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
    fig = figure(title)
    clf()
    axes = fig.subplots()
    axes.set_xlabel("Time in ms")
    axes.set_ylabel("Voltage in μV")
    channel_lines = Array{ChannelLine, 1}(undef, num_channels)
    for i = 1:num_channels
        line = plot([], [])[1]
        channel_lines[i] = ChannelLine(line)
    end
    return Plot(fig, axes, channel_lines, device, shown_channels, interval)
end

function update_plot!(plot::Plot)
    for c = 1:length(plot.channel_lines)
        state_plot = plot.channel_lines[c]
    end
end

device = Device(EEG3000([0, 0, 0, 0, 0]))
eeg_plot = Plot(device)

end