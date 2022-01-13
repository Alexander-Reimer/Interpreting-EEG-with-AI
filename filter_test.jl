using BrainFlow
using PyPlot

function plot_data(d, fig)
    figure(fig)
    clf()
    plot(d[:, 1])
    plot(d[:, 2])
end

function plot_data(d::Vector{Vector{Complex}}, fig)
    figure(fig)
    clf()
    plot(abs.(d[1]))
    plot(abs.(d[2]))
end

function power_2(d)
    return d[1:128]
end

d1_raw = BrainFlow.read_file("Blink/1.csv")

blink_raw = figure("Blink unfiltered")
plot_data(d1_raw, blink_raw)



d1_filt = deepcopy(d1_raw)
#d1_filt_c1 = deepcopy(d1_filt[:, 1])
#d1_filt_c2 = deepcopy(d1_filt[:, 2])

BrainFlow.remove_environmental_noise(view(d1_filt, :, 1), 200, BrainFlow.FIFTY)
BrainFlow.remove_environmental_noise(view(d1_filt, :, 2), 200, BrainFlow.FIFTY)

#=
BrainFlow.remove_environmental_noise(d1_filt_c1, 200, BrainFlow.FIFTY)
BrainFlow.remove_environmental_noise(d1_filt_c2, 200, BrainFlow.FIFTY)
=#

blink_filt = figure("Blink filtered")
plot_data(d1_filt, blink_filt)
#=
clf()
plot(d1_filt_c1)
plot(d1_filt_c2)
=#


d1_fft_raw = deepcopy(d1_raw)
d1_fft_raw = [BrainFlow.perform_fft(power_2(d1_fft_raw[:, i]), BrainFlow.NO_WINDOW) for i = 1:2]

blink_fft_raw = figure("Blink unfiltered FFT")
plot_data(d1_fft_raw, blink_fft_raw)


d1_fft_filt = deepcopy(d1_filt)
d1_fft_filt = [BrainFlow.perform_fft(power_2(d1_fft_filt[:, i]), BrainFlow.NO_WINDOW) for i = 1:2]

blink_fft_filt = figure("Blink filtered FFT")
plot_data(d1_fft_filt, blink_fft_filt)



d2_raw = BrainFlow.read_file("NoBlink/1.csv")

noBlink_raw = figure("NoBlink unfiltered")
plot_data(d2_raw, noBlink_raw)


d2_filt = deepcopy(d2_raw)
#d2_filt_c1 = deepcopy(d2_filt[:, 1])
#d2_filt_c2 = deepcopy(d2_filt[:, 2])

BrainFlow.remove_environmental_noise(view(d2_filt, :, 1), 200, BrainFlow.FIFTY)
BrainFlow.remove_environmental_noise(view(d2_filt, :, 2), 200, BrainFlow.FIFTY)

#=
BrainFlow.remove_environmental_noise(d2_filt_c1, 200, BrainFlow.FIFTY)
BrainFlow.remove_environmental_noise(d2_filt_c2, 200, BrainFlow.FIFTY)
=#

noBlink_filt = figure("NoBlink filtered")
plot_data(d2_filt, noBlink_filt)
#=
clf()
plot(d2_filt_c1)
plot(d2_filt_c2)
=#


d2_fft_raw = deepcopy(d2_raw)
d2_fft_raw = [BrainFlow.perform_fft(power_2(d2_fft_raw[:, i]), BrainFlow.NO_WINDOW) for i = 1:2]

noBlink_fft_raw = figure("NoBlink unfiltered FFT")
plot_data(d2_fft_raw, noBlink_fft_raw)


d2_fft_filt = deepcopy(d2_filt)
d2_fft_filt = [BrainFlow.perform_fft(power_2(d2_fft_filt[:, i]), BrainFlow.NO_WINDOW) for i = 1:2]

noBlink_fft_filt = figure("NoBlink filtered FFT")
plot_data(d2_fft_filt, noBlink_fft_filt)
