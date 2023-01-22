# module EEG
export Device
export Data
export Experiment, gather_data!, save_data, load_data, load_data!

"""
Abstract class containing data processors.

Defined data processors:
`StandardProcessor`: The standard processors with preset arguments and functions, for 
details see [`StandardProcessor`](@ref).
"""
abstract type DataProcessor end
abstract type DataDescriptor end
using BrainFlow, DataFrames, Dates, CSV, BSON, LSL
import PyPlot
Plt = PyPlot
Plt.pygui(true)

include("EEG_Devices.jl")

#= struct EEGDataType
    name::Symbol
end

RAW_EEG_DATA = EEGDataType(:raw_eeg)
FFT_DATA = EEGDataType(:raw_eeg) =#

# Utilitiy functions:
datetime(seconds_since_epoch) = Dates.unix2datetime(seconds_since_epoch)

"""
Device for gathering EEG data. Create it using
    
Device(board::EEGBoard)

... TODO
"""
mutable struct Device
    board::EEGBoard
    session_start::Float64
end

function Device(board::EEGBoard)
    return Device(board, time())
end

"""
Standard configuration for processing EEG data. It uses a preset of functions and 
options and may not work for you.

Create with [`Standard`](@ref).
"""
struct StandardProcessor <: DataProcessor
    output_type::Dict{Symbol,Any}
    fft_maxfreq::Integer
end
"""
    Standard()::StandardProcessor

Create standard configuration for processing EEG data. See [`StandardProcessor`](@ref)
for more details.
"""
function Standard()
    output_type = Dict()
    # requires output dims (num_channels, max_freq)
    output_type[:type] = :SpectralDensity
    processor = StandardProcessor(output_type, 60)
    return processor
end

"""
    process(device::Device, processor::StandardProcessor)

TODO: Processing

    1. Artifacts Removal
    2. Data Filtering
    3. Feature Extraction (FFT)
Also see [this](https://www.sciencedirect.com/science/article/pii\
/S1877705812022114/pdf?md5=3c7a2bdf5717d518cf46c4ef5d145d33&pid=1-s2\
.0-S1877705812022114-main.pdf), page 2529.
"""
function process(device::Device, processor::StandardProcessor)
    result = Array{Float64,2}(undef, device.board.num_channels, processor.fft_maxfreq)
    # Temporarily skip all processing
    # TODO
    for i in eachindex(result)
        result[i] = rand()
    end
    return result
end

struct Metadata
    md::Dict
end

function Base.copy(metadata::Metadata)
    new_md = copy(metadata.md)
    return Metadata(new_md)
end
# For `meta.name` syntax (e.g. meta.num_channels)
function Base.getproperty(meta::Metadata, name::Symbol)
    if name == :md
        return getfield(meta, name)
    else
        meta.md[name]
    end
end

# For `meta.name = value` syntax
function Base.setproperty!(meta::Metadata, name::Symbol, value)
    if name == :md
        setfield!(meta, name, value)
    else
        meta.md[name] = value
    end
end

"""
    _write_metadata(filepath, metadata::Metadata)

Internal method used for writing to file; in function to make later 
switch of file format easier.
"""
function _write_metadata(filepath, metadata::Metadata)
    if !isfile(filepath)
        createpath(filepath)
    end
    bson(filepath, metadata.md)
end

"""
    _read_metadata(filepath)

Internal method used for reading from file; in function to make later 
switch of file format easier.
"""
function _read_metadata(filepath)
    md = BSON.load(filepath, @__MODULE__)
    return Metadata(md)
end

mutable struct Data
    df::DataFrame
    metadata::Metadata
    name::String
    savedindex::Integer
    data_descriptor::DataDescriptor
end

"""
    create_data(name::String, data_desc::RawDataDescriptor)

Create new `Data`-Object for raw data (MCP3208).
"""
function create_data(name::String, data_desc::RawDataDescriptor)
    column_names = ["time", "tags", "extraInfo"]
    for channel = 1:data_desc.num_channels
        push!(column_names, string(channel))
    end
    types = fill(Float64[], data_desc.num_channels)
    #        Time       tags                      extra_info    voltages  
    types = [Float64[], Array{Array{String},1}(), Dict[], types...]
    df = DataFrame(types, column_names)
    # column_names =  ["version",     "num_samples",  "column_names"]
    # types =         [VersionNumber[], Integer[],     AbstractArray{String}[]]
    metadata = Metadata(Dict(
        :version => v"0.0.0",
        :num_samples => 0,
        :column_names => column_names,
        :descriptor => data_desc
    ))
    return Data(df, metadata, name, 0, data_desc)
end

"""
    create_data(name::String, data_desc::RawDataDescriptor)

Create new `Data`-Object for FFT data. 
"""
function create_data(name::String, data_desc::FFTDataDescriptor)
    column_names = ["time", "tags", "extraInfo"]
    for channel = 1:data_desc.num_channels
        for freq = 1:data_desc.max_freq
            push!(column_names, string(channel) * "_" * string(freq))
        end
    end
    types = fill(Float64[], data_desc.sample_width)
    #        Time       tags                      extra_info    voltages  
    types = [Float64[], Array{Array{String},1}(), Dict[], types...]
    df = DataFrame(types, column_names)
    # column_names =  ["version",       "num_samples", "column_names"]
    # types =         [VersionNumber[], Integer[],     AbstractArray{String}[]]
    metadata = Metadata(Dict(
        :version => v"0.0.0",
        :num_samples => 0,
        :column_names => column_names,
        :descriptor => data_desc
    ))
    return Data(df, metadata, name, 0, data_desc)
end

"""
    create_data(name::String, device::Device)

Create `Data`-Object which fits given device (raw data, fft data, etc.).
"""
function create_data(name::String, device::Device)
    return create_data(name, device.board.data_descriptor)
end

Base.push!(data::Data, val) = push!(data.df, val)

function Base.getproperty(data::Data, name::Symbol)
    if name in fieldnames(Data)
        return getfield(data, name)
    end
    return getproperty(data.metadata, name)
end

get_datapath(folderpath, name) = joinpath(folderpath, name * ".csv")

struct Experiment
    device::Device
    data::Data
    tags::Array{String,1}
    extra_info::Dict{Symbol,Any}
    folderpath::String
    # cases::Array{Symbol,1}
end

# For `experiment.num_samples`
function Base.getproperty(experiment::Experiment, name::Symbol)
    if name == :num_samples
        return experiment.data.metadata.num_samples
    else
        return getfield(experiment, name)
    end
end

"""
    Experiment(device::Device, name::String; tags::Array=[], 
    extra_info::Dict=Dict(), path::String="data/", load_previous::Bool=false)

`name`: Name of the experiment (e.g. "BlinkDetection").

`tags`: Tags which will be applied to all data gathered with this Experiment.

`path`: To top-level of data directory (e.g. "data/"). If empty, files can't be saved.

TODO: descs for keywords
TODO: `load_previous` not implemented yet (maybe in another function?)
"""
function Experiment(device::Device, name::String; tags::Array=[],
    extra_info::Dict=Dict(), path::String="data/", load_previous::Bool=false)
    folderpath = joinpath(path, name, "")
    data = create_data("RawData", device)
    experiment = Experiment(device, data, string.(tags), extra_info, folderpath)
    save_data(experiment)
    return experiment
end

"""
    clear!(data::Data)

Delete all saved data.
"""
function clear!(data::Data)
    empty!(data.df)
    data.metadata.num_samples = 0
    data.savedindex = 0
end

"""
    clear!(experiment::Experiment)

Delete all saved raw data from experiment.
"""
function clear!(experiment::Experiment)
    clear!(experiment.data)
end

"""
    get_sample!(board::EEGBoard)

Updates board.sample to new data from board.
"""
function get_sample!(board::EEGBoard)
    for channel = 1:board.data_descriptor.num_channels
        board.sample[channel] = get_voltage(board, channel)
    end
end

"""
    gather_data!(experiment::Experiment, runtime::Number; tags::Array=[], 
    extra_info::Dict=Dict())

Gather raw EEG data. 

`runtime`: Runtime in seconds.

Optional arguments:

`tags`: Tags to add to every data point on top of tags given to [`Experiment`](@ref)

`extra_info`: Extra info to add to every data point on top of extra info given 
to [`Experiment`](@ref)
"""
function gather_data!(experiment::Experiment, runtime::Number; tags::Array=[],
    extra_info::Dict=Dict(), delay::Number=0, save::Bool=true
)
    if !iscompatible(experiment.data, load_metadata(experiment))
        throw(ErrorException("TODO")) # TODO
    end

    start_time = time()

    new_row = Array{Any,1}(
        undef, 3 + experiment.device.board.data_descriptor.sample_width
    )

    # combine default strings with given strings
    new_row[2] = vcat(experiment.tags, string.(tags))
    new_row[3] = merge(experiment.extra_info, extra_info)

    while (time() - start_time) < runtime
        new_row[1] = time()
        # update sample stored in `device`
        get_sample!(experiment.device.board)
        # bring this data into new_row
        new_row[4:end] = experiment.device.board.sample
        push!(experiment.data, new_row)

        if save
            save_data(experiment, updatemeta=false, checkmeta=false)
        end
        if delay != 0
            sleep(delay)
        end
        experiment.data.metadata.num_samples += 1
    end
end

get_metadatapath(folder_path, name) = joinpath(folder_path, name * "Metadata.bson")

function load_metadata(path::String)
    if isfile(path)
        return _read_metadata(path)
    else
        return nothing
    end
end

load_metadata(folder_path, name) = load_metadata(get_metadatapath(folder_path, name))
load_metadata(experiment::Experiment) = load_metadata(experiment.folderpath, "RawData")

"""
    is_compat(df::DataFrame, metadata::Union{Metadata, Nothing})

Check if `df` is "compatible" with `metadata`.

If `metadata` is `nothing`, then return true (for easy use with `load_metadata`
which returns nothing when no `metadata` is defined)
"""
function iscompatible(df::DataFrame, metadata::Union{Metadata,Nothing})
    if metadata === nothing
        return true
    end
    if metadata.column_names == names(df)
        return true
    end
    return false
end

iscompatible(data::Data, metadata::Union{Metadata,Nothing}) = iscompatible(
    data.df,
    metadata
)

function combine_metadata(data::Data, meta::Metadata; repeat=false)
    # TODO: weird?
    new_meta = copy(meta)
    add_length = data.metadata.num_samples
    if !repeat
        add_length -= data.savedindex
    end
    new_meta.num_samples += add_length
    return new_meta
end

"""
    create_path(path::String)

Create necessary folders and file if they don't exist yet, so that
`isdir(path)` or `isfile(path)` returns true depending on whether
`path` points to a folder or a file.
"""
function createpath(path::String)
    mkpath(dirname(path)) # create directories containing file
    if !isfile(path)
        io = open(path, create=true) # create file
        close(io)
    end
end

function save_data(data::Data, df::DataFrame, folderpath; checkmeta=true,
    updatemeta=true, repeat=false)
    datapath = joinpath(folderpath, data.name * ".csv")
    metapath = get_metadatapath(folderpath, data.name)
    metadata = nothing
    if checkmeta && isfile(metapath)
        if !iscompatible(data.df, load_metadata(metapath))
            throw(ErrorException( # TODO: more specific error
                "Data not compatible with DataFrame at $file_path 
                (perhaps processed vs non-processed data or different 
                number of channels). If you want to overwrite the old data,
                delete both $file_path and $meta_file_path."))
        end
    end
    if updatemeta
        # TODO: what?
        if metadata === nothing
            metadata = load_metadata(metapath)
        end
        if metadata === nothing
            newmeta = data.metadata
        else
            newmeta = combine_metadata(data, metadata)
        end
        _write_metadata(metapath, newmeta)
    end
    # Make sure file exist
    overwrite = false
    if !isfile(datapath)
        createpath(datapath)
        overwrite = true # overwrite to create headers
    end
    # deactivate append if overwrite --> file is overwritten
    CSV.write(datapath, df, append=!overwrite)
end

function save_data(data::Data, folderpath; checkmeta=true, updatemeta=true,
    repeat=false)
    # if !isempty(data.df)
    if repeat
        df_new = data.df
    else
        df_new = data.df[(data.savedindex+1):end, :]
    end
    save_data(data, df_new, folderpath, checkmeta=checkmeta, updatemeta=updatemeta,
        repeat=repeat)
    data.savedindex = lastindex(data.df, 1)
    # end
end

function save_data(experiment::Experiment; checkmeta=true, updatemeta=true,
    repeat=false)
    save_data(experiment.data, experiment.folderpath, checkmeta=checkmeta,
        updatemeta=updatemeta, repeat=repeat)
end

function CSV.tryparse(t::Type{Vector{String}}, str::String)
    return chop.(split(chop(str, head=1, tail=1), ", "), head=1, tail=1)
end

# function CSV.tryparse(t::Type{Dict{Any, Any}}, str::String)
#     return chop.(split(chop(str, head=1, tail=1), ", "), head=1, tail=1)
# end

function load_data(folderpath, name; start_pos=1, num_samples=:all, exact_num=false)
    metapath = get_metadatapath(folderpath, name)
    metadata = load_metadata(metapath) # TODO: loading RawData not working because of meta name
    if metadata === nothing
        throw(ErrorException("Metadata from $metapath couldn't be read! 
        Maybe the file doesn't exist anymore?"))
    end

    datapath = get_datapath(folderpath, name)
    type_map = Dict(:tags => Array{String, 1})
    if num_samples == :all
        df = CSV.read(datapath, DataFrame; skipto=start_pos + 1, types=type_map) # +1 for headers
    else
        ntasks = exact_num ? 1 : Threads.nthreads()
        df = CSV.read(datapath, DataFrame; skipto=start_pos + 1, limit=num_samples,
            ntasks=ntasks, types=type_map) # +1 for headers
    end

    # Update metadata in case data was previously saved using updatemeta = false
    metadata.num_samples = size(df, 1)
    # And save it again
    _write_metadata(metapath, metadata)

    data = create_data(name, metadata.descriptor)
    data.metadata = metadata
    data.df = df
    data.savedindex = data.num_samples
    return data
end

# TODO: test
# TODO: use savedindex to avoid duplicates
function combine!(data1::Data, data2::Data; append=true)
    if append
        append!(data1.df, data2.df, promote=true)
    else
        prepend!(data1.df, data2.df, promote=true)
    end
    data1.metadata = combine_metadata(data2, data1.metadata, repeat=true)
end

# TODO: test
function load_data!(experiment::Experiment; start_pos=1, num_samples=:all,
    exact_num=false, append=true)
    old_data = load_data(experiment.folderpath, experiment.data.name,
        start_pos=start_pos, num_samples=num_samples, exact_num=exact_num)
    combine!(experiment.data, old_data, append=append)
end

mutable struct DataFilter
    include_tags::Array{String,1}
    exclude_tags::Array{String,1}
    extra_info_filter::Function
end

struct DataHandler
    data_processor::DataProcessor
    data_io#::DataIO
    cases::Union{Array{Symbol,1},Nothing}
    name::Union{String,Nothing}
    max_freq::Union{Int,Nothing}
end

"""
    DataHandler(data_processor::DataProcessor, data_io::DataIO; cases=nothing,
    name=nothing, max_freq=nothing)

Create a DataHandler instance. `cases`, `name` and `max_freq` are automatically 
determined by data saved at path if they are `nothing`.

Example:

    data_io = DataIO("data/test", states)
    data_handler = DataHandler(data_io, StandardFFT())
"""
function DataHandler(data_processor::DataProcessor, data_io; cases=nothing,
    name=nothing, max_freq=nothing)

end
# end