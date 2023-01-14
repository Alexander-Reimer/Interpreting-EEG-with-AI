module BCIInterface
include("EEG.jl")
# using .EEG
# import .EEG: *
include("AI_new.jl")
export MCP3208, GanglionGUI, Device
export Experiment, gather_data!, save_data, load_data, load_data!
# Write your package code here.

end
