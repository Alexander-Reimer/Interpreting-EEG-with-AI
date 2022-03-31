using PyPlot
using BrainFlow

function plot_eeg_sample(path, channels = [1, 2, 3, 4])
    fig = figure("EEG")
    xlabel("Zeit in Millisekunden")
    ylabel("Spannungsdifferenz in Mikrovolt")
    x = [i for i = 5:5:1000]
    data = BrainFlow.read_file(path)
    for channel in channels
        plot(x, data[:, channel], label = "Elektrode $channel")
    end
    legend(loc="best")
    fig.tight_layout()
end