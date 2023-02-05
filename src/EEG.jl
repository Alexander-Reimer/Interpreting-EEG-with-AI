include("EEG_Devices.jl")

"""
Abstract class containing data processors.

Defined data processors:

`StandardProcessor`: The standard processors with preset arguments and functions, for
details see [`StandardProcessor`](@ref).
"""
abstract type AbstractDataProcessor end

"""
    datetime(seconds_since_epoch)

Convert seconds since unix epoch into a DateTime object.
"""
datetime(seconds_since_epoch::Number) = Dates.unix2datetime(seconds_since_epoch)

"""
Standard configuration for processing EEG data. It uses a preset of functions and
options and may not work for you.

Create with [`Standard`](@ref).
"""
struct StandardProcessor <: AbstractDataProcessor
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
    process(board::EEGBoard, processor::StandardProcessor)

TODO: Processing

    1. Artifacts Removal
    2. Data Filtering
    3. Feature Extraction (FFT)
Also see [this](https://www.sciencedirect.com/science/article/pii\
/S1877705812022114/pdf?md5=3c7a2bdf5717d518cf46c4ef5d145d33&pid=1-s2\
.0-S1877705812022114-main.pdf), page 2529.
"""
function process(board::EEGBoard, processor::StandardProcessor)
    result = Array{Float64,2}(undef, board.num_channels, processor.fft_maxfreq)
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

Base.:(==)(a::Metadata, b::Metadata) = a.md == b.md
Base.copy(metadata::Metadata) = Metadata(copy(metadata.md))
# For `meta.name` syntax (e.g. meta.num_channels)
Base.getproperty(x::Metadata, name::Symbol) = name == :md ? getfield(x, name) : x.md[name]
# For `meta.name = value` syntax
function Base.setproperty!(x::Metadata, name::Symbol, value)
    return name == :md ? setfield!(x, name, value) : x.md[name] = value
end

"""
    _write_metadata(filepath, metadata::Metadata)

Internal method used for writing to file; in function to make later
switch of file format easier.
"""
function _write_metadata(path, metadata::Metadata)
    if !isfile(path)
        createpath(path)
    end
    return bson(path, metadata.md)
end

"""
    _read_metadata(filepath)

Internal method used for reading from file; in function to make later
switch of file format easier.
"""
function _read_metadata(path)
    md = BSON.load(path, @__MODULE__)
    return Metadata(md)
end

load_metadata(path::String) = isfile(path) ? _read_metadata(path) : nothing
get_metadatapath(folder, name) = joinpath(folder, name * "Metadata.bson")
load_metadata(folder, name) = load_metadata(get_metadatapath(folder, name))

# TODO: clarify "compatible"
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

mutable struct Data
    df::DataFrame
    metadata::Metadata
    name::String
    dir::String
    savedindex::Integer
    data_descriptor::AbstractDataDescriptor
end

function Base.copy(x::Data)
    return Data(
        copy(x.df), copy(x.metadata), x.name, x.dir, x.savedindex, copy(x.data_descriptor)
    )
end
function Base.:(==)(a::Data, b::Data)
    return a.df == b.df &&
           a.metadata == b.metadata &&
           a.name == b.name &&
           a.dir == b.dir &&
           a.savedindex == b.savedindex &&
           a.data_descriptor == b.data_descriptor
end
Base.push!(data::Data, val) = push!(data.df, val)
function Base.getproperty(x::Data, name::Symbol)
    if name == :folderpath
        return joinpath(x.dir, x.name)
    elseif name in fieldnames(Data)
        return getfield(x, name)
    else
        return getproperty(x.metadata, name)
    end
end

load_metadata(data::Data; name="data") = load_metadata(joinpath(data.dir, data.name), name)

"""
    Data(name::String, data_desc::RawDataDescriptor; dir = "data/")

Create new `Data`-Object for raw data (MCP3208).
"""
function Data(name::String, data_desc::RawDataDescriptor; dir="data/")
    column_names = ["time", "tags", "extraInfo"]
    for channel in 1:(data_desc.num_channels)
        push!(column_names, string(channel))
    end
    types = fill(Float64[], data_desc.num_channels)
    #        Time       tags                      extra_info    voltages
    types = [Float64[], Array{Array{String},1}(), Dict[], types...]
    df = DataFrame(types, column_names)
    # column_names =  ["version",     "num_samples",  "column_names"]
    # types =         [VersionNumber[], Integer[],     AbstractArray{String}[]]
    metadata = Metadata(
        Dict(
            :version => v"0.0.0",
            :num_samples => 0,
            :column_names => column_names,
            :descriptor => data_desc,
        ),
    )
    return Data(df, metadata, name, dir, 0, data_desc)
end

"""
    Data(name::String, data_desc::FFTDataDescriptor; dir="data/")

Create new `Data`-Object for FFT data.
"""
function Data(name::String, data_desc::FFTDataDescriptor; dir="data/")
    column_names = ["time", "tags", "extraInfo"]
    for channel in 1:(data_desc.num_channels)
        for freq in 1:(data_desc.max_freq)
            push!(column_names, string(channel) * "_" * string(freq))
        end
    end
    types = fill(Float64[], data_desc.sample_width)
    #        Time       tags                      extra_info    voltages
    types = [Float64[], Array{Array{String},1}(), Dict[], types...]
    df = DataFrame(types, column_names)
    # column_names =  ["version",       "num_samples", "column_names"]
    # types =         [VersionNumber[], Integer[],     AbstractArray{String}[]]
    metadata = Metadata(
        Dict(
            :version => v"0.0.0",
            :num_samples => 0,
            :column_names => column_names,
            :descriptor => data_desc,
        ),
    )
    return Data(df, metadata, name, dir, 0, data_desc)
end

"""
    Data(name::String, board::EEGBoard; dir="data/")

Create `Data`-Object which fits given board (raw data, fft data, etc.).
"""
function Data(name::String, board::EEGBoard; dir="data/")
    return Data(name, board.data_descriptor; dir=dir)
end

"""
    clear!(data::Data)

Delete all saved data.
"""
function clear!(data::Data)
    empty!(data.df)
    data.metadata.num_samples = 0
    return data.savedindex = 0
end

get_datapath(folder, name) = joinpath(folder, name * ".csv")
iscompatible(d::Data, metadata::Union{Metadata,Nothing}) = iscompatible(d.df, metadata)

# TODO: doc string
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
    # if path points to file
    if !isdirpath(path)
        # check if already exists
        if !isfile(path)
            io = open(path; create=true) # create file
            close(io)
        end
    end
end

"""
    save(data::Data, df::DataFrame; name="data", checkmeta=true, updatemeta=true,
        overwrite=false)

Save given `df`. Folder is determined by `data` (`data.dir` and `data.name`), file name by
`name`.

# Keywords
- `checkmeta=true`: Load and check `Metadata` saved at the location for compatibility if it
  already exists.

- `updatemeta=true`: Save the updated `Metadata`.

- `overwrite=false`: When true, overwrite already existing data and meta data.
"""
function save(
    data::Data, df::DataFrame; name="data", checkmeta=true, updatemeta=true, overwrite=false
)
    dir = joinpath(data.dir, data.name)
    datapath = get_datapath(dir, name)
    metapath = get_metadatapath(dir, name)
    # only load metadata if it is needed; check compatibility even if checkmeta is false
    if checkmeta
        metadata = load_metadata(metapath)
        if !iscompatible(data.df, metadata)
            throw(
                ErrorException( # TODO: more specific error
                    "Data not compatible with DataFrame at $file_path
                    (perhaps processed vs non-processed data or different
                    number of channels). If you want to overwrite the old data,
                    delete both $datapath and $metapath.",
                )
            )
        end
    end
    if updatemeta
        if metadata === nothing || overwrite
            # if metadata doesn't exist yet or should be overwritten
            newmeta = data.metadata
        else
            newmeta = combine_metadata(data, metadata)
        end
        _write_metadata(metapath, newmeta)
    end
    append = !overwrite
    # Make sure file exist
    if !isfile(datapath)
        createpath(datapath)
        # if file doesn't exist yet, append must be false as otherwise, headers won't be
        # written
        append = false
    end
    # TODO: also check file contents if header exists.
    return CSV.write(datapath, df; append=append)
end

"""
    save(data::Data; checkmeta=true, updatemeta=true, overwrite=false, repeat=false)

Save given `data`. Folder is determined by `data` (`data.dir * "/" * data.name`), file name by
`name`.
"""
function save(
    data::Data; name="data", checkmeta=true, updatemeta=true, overwrite=false, repeat=false
)
    if repeat
        df_new = data.df
    else
        df_new = data.df[(data.savedindex + 1):end, :]
    end
    save(
        data,
        df_new;
        name=name,
        checkmeta=checkmeta,
        updatemeta=updatemeta,
        overwrite=overwrite,
    )
    return data.savedindex = lastindex(data.df, 1)
end

struct Experiment
    board::EEGBoard
    data::Data
    tags::Array{String,1}
    extra_info::Dict{Symbol,Any}
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
    Experiment(board::EEGBoard, name::String; tags::Array=[],
    extra_info::Dict= Dict(), path::String="data/", load_previous::Bool=false)

`name`: Name of the experiment (e.g. "BlinkDetection").

`tags`: Tags which will be applied to all data gathered with this Experiment.

`path`: To top-level of data directory (e.g. "data/"). If empty, files can't be saved.

TODO: descs for keywords
"""
function Experiment(
    board::EEGBoard,
    name::String;
    tags::Array=[],
    extra_info::Dict=Dict(),
    dir::String="data/",
)
    data = Data(name, board; dir=dir)
    experiment = Experiment(board, data, string.(tags), extra_info)
    save(experiment)
    return experiment
end

"""
    clear!(experiment::Experiment)

Delete all saved data.
"""
clear!(experiment::Experiment) = clear!(experiment.data)

# load_metadata(experiment::Experiment) = load_metadata(experiment.data)
function load_metadata(experiment::Experiment, name="data")
    return load_metadata(experiment.data; name=name)
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
function gather_data!(
    experiment::Experiment,
    runtime::Number;
    tags::Array=[],
    extra_info::Dict=Dict(),
    delay::Number=0,
    autosave::Bool=true,
    name::String="data",
)
    if !iscompatible(experiment.data, load_metadata(experiment))
        throw(ErrorException("TODO")) # TODO
    end

    new_row = Array{Any,1}(undef, 3 + experiment.board.data_descriptor.sample_width)
    # combine default strings with given strings
    new_row[2] = vcat(experiment.tags, string.(tags))
    new_row[3] = merge(experiment.extra_info, extra_info)

    start_time = time()
    while (time() - start_time) < runtime
        new_row[1] = time()
        # update sample stored in `board`
        get_sample!(experiment.board)
        # bring this data into new_row
        new_row[4:end] = experiment.board.sample
        push!(experiment.data, new_row)

        if autosave
            save(experiment; updatemeta=false, checkmeta=false)
        end
        if delay != 0
            sleep(delay)
        end
        experiment.data.metadata.num_samples += 1 # TODO
    end
    return save(experiment)
end

function save(
    experiment::Experiment; name="data", checkmeta=true, updatemeta=true, repeat=false
)
    return save(
        experiment.data;
        name=name,
        checkmeta=checkmeta,
        updatemeta=updatemeta,
        repeat=repeat,
    )
end

function CSV.tryparse(t::Type{Vector{String}}, str::String)
    return chop.(split(chop(str; head=1, tail=1), ", "), head=1, tail=1)
end

# function CSV.tryparse(t::Type{Dict{Any, Any}}, str::String)
#     return chop.(split(chop(str, head=1, tail=1), ", "), head=1, tail=1)
# end

function load_data(
    dataname; dir="data", name="data", start_pos=1, num_samples=:all, exact_num=false
)
    folderpath = joinpath(dir, dataname)
    metapath = get_metadatapath(folderpath, name)
    # TODO: loading RawData not working because of meta name
    metadata = load_metadata(metapath)
    if metadata === nothing
        throw(ErrorException("Metadata from $metapath couldn't be read!
        Maybe the file doesn't exist anymore?"))
    end

    datapath = get_datapath(folderpath, name)
    type_map = Dict(:tags => Array{String,1}, :time => Float64) #:extraInfo=>Dict{Any,Any}
    if num_samples == :all
        # +1 for headers
        df = CSV.read(datapath, DataFrame; skipto=start_pos + 1, types=type_map)
    else
        ntasks = exact_num ? 1 : Threads.nthreads()
        df = CSV.read(
            datapath,
            DataFrame;
            skipto=start_pos + 1,
            limit=num_samples,
            ntasks=ntasks,
            types=type_map,
        ) # +1 for headers
    end

    # Update metadata in case data was previously saved using updatemeta = false
    metadata.num_samples = size(df, 1)
    # And save it again
    _write_metadata(metapath, metadata)

    data = Data(dataname, metadata.descriptor; dir=dir)
    data.metadata = metadata
    data.df = df
    data.savedindex = data.num_samples
    return data
end

# TODO: test
# TODO: use savedindex to avoid duplicates
function combine!(data1::Data, data2::Data; append=true)
    if append
        append!(data1.df, data2.df; promote=true)
    else
        prepend!(data1.df, data2.df; promote=true)
    end
    return data1.metadata = combine_metadata(data2, data1.metadata; repeat=true)
end

function load_data!(
    experiment::Experiment;
    name::String="data",
    start_pos=1,
    num_samples=:all,
    exact_num=false,
    repeat=false,
    append=true,
)
    if !repeat
        if start_pos !== 1
            @warn ("Given start_pos will be ignored because repeat=true!")
        end
        start_pos = experiment.data.savedindex + 1
    end

    old_data = load_data(
        experiment.data.name;
        dir=experiment.data.dir,
        name=name,
        start_pos=start_pos,
        num_samples=num_samples,
        exact_num=exact_num,
    )
    return combine!(experiment.data, old_data; append=append)
end

mutable struct DataFilter
    include_tags::Array{String,1}
    exclude_tags::Array{String,1}
    extra_info_filter::Function
end

struct DataHandler
    data_processor::AbstractDataProcessor
    data_io::Any#::DataIO
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
function DataHandler(
    data_processor::AbstractDataProcessor,
    data_io;
    cases=nothing,
    name=nothing,
    max_freq=nothing,
) end
