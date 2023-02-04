module BCIInterface

using BaremetalPi # used: init_spi, spi_transfer!
using BSON # used: bson, load
using CSV # used: write, read, extended: tryparse
using CUDA # used: functional
using DataFrames # used: DataFrame, empty!, SubDataFrame
using Dates # used: now, format
# used: Adam, Conv, Chain, cpu, DataLoader, Dense, Dropout, flatten, gpu, logitcrossentropy,
# setup, throttle, update!, withgradient
using Flux
using LSL # used: channel_count, pull_sample, pull_sample!, resolve_streams, StreamInlet
using ProgressMeter # used: Progress, update!
using PyPlot # used: pygui

include("EEG.jl")
include("AI.jl")

Plt = PyPlot
Plt.pygui(true)

# EEG
export Device, Data, Experiment, gather_data!, save, load_data, load_data!
export MCP3208, GanglionGUI
# AI
export ModelData, create_model, train!, save, load_model, parse_modelname
export cpu, gpu

end
