# Gathering, processing and loading EEG data
## Introduction

## Gathering data
To gather EEG data, you will first need a physical EEG device. We are currently developing our own which you can build yourself at home with the instrucations in [The MCP3208 -- a cheap alternative for commercial EEGs](@ref).

This framework currently only offers direct support for this EEG. However, support can be easily extended, for instructions see [Using your own EEG device](@ref).

The following steps assume that you already have the hardware set up. If you have added your own EEG, just replace `MCP3208` with the custom function you created.

To gather EEG data, you need to create a `Device` object with `Device(board::EEGBoard)`. An Example:
```julia
device = Device(MCP3208("/dev/spidev0.0", 8))
```

For details about `MCP3208`, see [`MCP3208`](@ref).
## Processing data

## Reference
```@index
Modules = [BCIInterface.EEG]
```
```@autodocs
Modules = [BCIInterface.EEG]
```