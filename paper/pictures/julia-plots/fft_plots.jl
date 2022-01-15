using Pkg
cur_path = pwd()
cd("../../..")
Pkg.activate(".")
cd(cur_path)
using PyPlot
using FFTW

#=
function add_frequency!(x, y, frq)
    y .+= sin.(2π * x * frq)
end

function frequency(len, sample_rate, freqs)
    stepsize = 1 / sample_rate
    x = [i for i = 0:stepsize:len]
    y = zeros(length(x))
    for frq in freqs
        add_frequency!(x, y, frq)
    end
    return x, y
end
=#

function get_fft(x, y; time = nothing)
    if time === nothing
        if x === nothing
            @error "Neither x not time specified!"
        else
            time = x[end]
        end
    end
    x_fft = [i for i = 0.0:length(y)-1]
    y_fft = abs.(fft(y))

    # Remove Mirror Effect
    len_fft = length(y_fft)
    half_len = round(Int, len_fft / 2)
    y_fft = y_fft[1:half_len]

    # Adjust to same length of y values
    x_fft = x_fft[1:half_len]

    # Adjust to duration
    x_fft ./= time
    y_fft ./= time


    return x_fft, y_fft
end

#=
function plot_pure_freqs_comparison()
    figure("Original")
    x, y = frequency(4, 1000, [2, 3, 7])
    plot(x, y)

    figure("FFT")
    x_f, y_f = get_fft(x, y)
    plot(x_f, y_f)
end
=#

function fft_eeg(eeg_data, time)
    x_fft, y_fft = get_fft(nothing, eeg_data, time = time)
    return x_fft, y_fft
end

function divide_into_channels(data, channels)
    split_data = []
    per_channel = Int(length(data) / channels)
    
    for i = 1:channels
        i_start = (i - 1) * per_channel + 1
        i_end = i * per_channel
        push!(split_data, data[i_start:i_end])
    end
    return split_data
end

function plot_eegs_fft()
    curr_path = pwd()
    cd("../../..")

    include("main.jl")
    sleep(0.2)
    global eeg_data = AI.get_loader()[1].data[1]
    diviser = Int(size(eeg_data)[2] / 2)
    blink_data = eeg_data[:, 1:diviser]
    no_blink_data = eeg_data[:, diviser+1:end]

    cd(curr_path)

    eeg_fft_fig = figure("Die FFTs der EEG-Daten")

    xlabel("Frequenz in Herz")

    # Channel 1

    # Blink:
    global current_data = divide_into_channels(blink_data[:, 1], 2)
    x, y = fft_eeg(current_data[1], 1)
    plot(x, y, "blue", label = "Geblinzelt")

    for sample_i = 2:size(blink_data)[2]
        global current_data = divide_into_channels(blink_data[:, sample_i], 2)
        x, y = fft_eeg(current_data[1], 1)
        plot(x, y, "blue")
    end

    # No blink:
    global current_data = divide_into_channels(no_blink_data[:, 1], 2)
    x, y = fft_eeg(current_data[1], 1)
    plot(x, y, "red", label="Nicht geblinzelt")

    for sample_i = 2:size(blink_data)[2]
        global current_data = divide_into_channels(no_blink_data[:, sample_i], 2)
        x, y = fft_eeg(current_data[1], 1)
        plot(x, y, "red")
    end

    legend()
    
    return x, y
end



#=
figure("Drei Sekunden, Original")
x, y = frequency(2, 1000, [2, 3, 7])
plot(x, y)

figure("Drei Sekunden, FFT")
x_f, y_f = get_fft(y)
plot(x_f, y_f)
=#