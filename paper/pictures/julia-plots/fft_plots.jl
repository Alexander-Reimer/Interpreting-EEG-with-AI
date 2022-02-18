using Pkg
cur_path = pwd()
cd("../../..")
Pkg.activate(".")
cd(cur_path)
using PyPlot
using FFTW


function add_frequency!(x, y, frq, amplitude = 1)
    y .+= amplitude .* sin.(2π * x * frq)
end

function frequency(len, sample_rate, freqs, amplitudes = nothing)
    if amplitudes === nothing
        fill(1, length(freqs))
    end
    stepsize = 1 / sample_rate
    x = [i for i = 0:stepsize:len]
    y = zeros(length(x))
    for i = 1:length(freqs)
        add_frequency!(x, y, freqs[i], amplitudes[i])
    end
    return x, y
end


function get_fft(x, y; time = nothing)
    if time === nothing
        if x === nothing
            @error "Neither x nor time specified!"
        else
            time = x[end]
        end
    end
    x_fft = [i for i = 0.0:length(y)-1]
    y_fft = (fft(y))

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
    #x_fft, y_fft = get_fft(nothing, eeg_data, time = time)
    y_fft = rfft(eeg_data)
    return y_fft
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

function get_eeg_data(path, data_x, data_y, endings, output)
    sample_number = 1
    while isfile(path * string(sample_number) * ".csv")
        # Read recorded EEG data
        sample_data_x = BrainFlow.read_file(path * string(sample_number) * ".csv")
        # Remove 3rd and 4th channel as they are a lot worse than 1st and 2nd and aren't necessary
        sample_data_x = sample_data_x[:, 1:2]
        # Filter 50 Hz frequencies to remove environmental noise
        for i = 1:size(sample_data_x)[2]
            BrainFlow.remove_environmental_noise(view(sample_data_x, :, i), 200, BrainFlow.FIFTY)
        end

        sample_data_x = reshape(sample_data_x, (:, 1))
        #sample_data_x[800] = endings[1][sample_number]

        # Perform FFT on data, once per channel
        #sample_data_x = [make_fft(sample_data_x[1:200])..., make_fft(sample_data_x[201:400])...]
        # Append to existing data
        data_x = [data_x sample_data_x]
        sample_number += 1
    end

    # Append given "output" to data_y ([1.0, 0.0] for Blink and [0.0, 1.0] for NoBlink)
    for i = 1:sample_number-1
        data_y = [data_y output]
    end

    return data_x, data_y
end

function own_ifft(x_fft, y_fft)
    
end

function blink_fft()
    path = "C:/Users/alexs/OneDrive/Dokumente/Programming/Interpreting-EEG-with-AI/Blink/first_samples-before_01-15-2022/"
    global data_x = Array{Float64}(undef, 400, 0)
    global data_y = Array{Float64}(undef, 2, 0)
    data_x, data_y = get_eeg_data(path, data_x, data_y, nothing, [1, 0])

    fig = figure("Blink FFT")
    plt

    current_d = data_x[:, 1]
    current_d = divide_into_channels(current_d, 2)

    ax1 = plt.subplot(131, title = "Originale EEG Daten")
    xlabel("Zeit in 5ms")
    ylabel("Spannungsdifferenz in Mikrovolt")
    #ax1.plot(current_d[1], color = "red")
    ax1.plot(current_d[2])

    ax2 = plt.subplot(132, title = "FFT (Spektralanalyse)")
    xlabel("Frequenz in Hertz")
    ylabel("Amplitude")
    y_fft_1 = fft_eeg(current_d[1], 1)
    global y_fft_2 = fft_eeg(current_d[2], 1)
    #ax2.plot(x_fft_1, y_fft_1, color = "red")
    ax2.plot(abs.(y_fft_2))


    ax3 = plt.subplot(133, title = "IFFT (Von FFT rekonstruiert)", sharex = ax1, sharey = ax1)
    xlabel("Zeit in 5ms")
    ylabel("Spannungsdifferenz in Mikrovolt")
    
    y_ifft_1 = irfft(y_fft_1, 200)
    y_ifft_2 = irfft(y_fft_2, 200)
    #ax2.plot(x_fft_1, y_fft_1, color = "red")
    ax3.plot(y_ifft_2)

end