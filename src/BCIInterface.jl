module BCIInterface
include("EEG.jl")
using .EEG
export MCP3208, Device
export Experiment, gather_data!, save_data, load_data, load_data!
# Write your package code here.

end
