var documenterSearchIndex = {"docs":
[{"location":"MCP3208/#The-MCP3208-–-a-cheap-alternative-for-commercial-EEGs","page":"MCP3208","title":"The MCP3208 – a cheap alternative for commercial EEGs","text":"","category":"section"},{"location":"eeg/#Gathering,-processing-and-loading-EEG-data","page":"EEG","title":"Gathering, processing and loading EEG data","text":"","category":"section"},{"location":"eeg/#Introduction","page":"EEG","title":"Introduction","text":"","category":"section"},{"location":"eeg/#Gathering-data","page":"EEG","title":"Gathering data","text":"","category":"section"},{"location":"eeg/","page":"EEG","title":"EEG","text":"To gather EEG data, you will first need a physical EEG device. We are currently developing our own which you can build yourself at home with the instrucations in The MCP3208 – a cheap alternative for commercial EEGs.","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"This framework currently only offers direct support for this EEG. However, support can be easily extended, for instructions see Using your own EEG device.","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"The following steps assume that you already have the hardware set up. If you have added your own EEG, just replace MCP3208 with the custom function you created.","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"To gather EEG data, you need to create a Device object with Device(board::EEGBoard). An Example:","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"device = Device(MCP3208(\"/dev/spidev0.0\", 8))","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"For details about MCP3208, see MCP3208.","category":"page"},{"location":"eeg/#Processing-data","page":"EEG","title":"Processing data","text":"","category":"section"},{"location":"eeg/#Reference","page":"EEG","title":"Reference","text":"","category":"section"},{"location":"eeg/","page":"EEG","title":"EEG","text":"Modules = [BCIInterface.EEG]","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"Modules = [BCIInterface.EEG]","category":"page"},{"location":"eeg/#BCIInterface.EEG.DataHandler-Tuple{BCIInterface.EEG.DataProcessor, Any}","page":"EEG","title":"BCIInterface.EEG.DataHandler","text":"DataHandler(data_processor::DataProcessor, data_io::DataIO; cases=nothing,\nname=nothing, max_freq=nothing)\n\nCreate a DataHandler instance. cases, name and max_freq are automatically  determined by data saved at path if they are nothing.\n\nExample:\n\ndata_io = DataIO(\"data/test\", states)\ndata_handler = DataHandler(data_io, StandardFFT())\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.DataProcessor","page":"EEG","title":"BCIInterface.EEG.DataProcessor","text":"Abstract class containing data processors.\n\nDefined data processors: StandardProcessor: The standard processors with preset arguments and functions, for  details see StandardProcessor.\n\n\n\n\n\n","category":"type"},{"location":"eeg/#BCIInterface.EEG.Device","page":"EEG","title":"BCIInterface.EEG.Device","text":"Device for gathering EEG data. Create it using\n\nDevice(board::EEGBoard)\n\n... TODO\n\n\n\n\n\n","category":"type"},{"location":"eeg/#BCIInterface.EEG.Experiment-Tuple{Device, String}","page":"EEG","title":"BCIInterface.EEG.Experiment","text":"name: Name of the experiment (e.g. \"BlinkDetection\").\n\ntags: Tags which will be applied to all data gathered with this Experiment.\n\npath: To top-level of data directory (e.g. \"data/\"). If empty, files can't be saved.\n\nTODO: load_previous not implemented yet (maybe in another function?)\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.MCP3208-Tuple{String, Int64}","page":"EEG","title":"BCIInterface.EEG.MCP3208","text":"MCP3208(path::String, num_channels::Int; max_speed_hz::Int=1000, \nonline=true)\n\nInitialise a MCP3208-based self-built EEG. path refers to the SPI  path of the device and num_channels to the number of connected  electrodes.\n\nWith online set to true, the EEG will be \"simulated\": get_voltage(...)  will return random values.\n\nTODO: maxspeedhz\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.StandardProcessor","page":"EEG","title":"BCIInterface.EEG.StandardProcessor","text":"Standard configuration for processing EEG data. It uses a preset of functions and  options and may not work for you.\n\nCreate with Standard.\n\n\n\n\n\n","category":"type"},{"location":"eeg/#BCIInterface.EEG.Standard-Tuple{}","page":"EEG","title":"BCIInterface.EEG.Standard","text":"Standard()::StandardProcessor\n\nCreate standard configuration for processing EEG data. See StandardProcessor for more details.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG._read_metadata-Tuple{Any}","page":"EEG","title":"BCIInterface.EEG._read_metadata","text":"Internal method used for reading from file; in function to make later  switch of file format easier.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG._write_metadata-Tuple{Any, BCIInterface.EEG.Metadata}","page":"EEG","title":"BCIInterface.EEG._write_metadata","text":"Internal method used for writing to file; in function to make later  switch of file format easier.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.clear!-Tuple{BCIInterface.EEG.Data}","page":"EEG","title":"BCIInterface.EEG.clear!","text":"clear!(data::Data)\n\nDelete all saved data.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.clear!-Tuple{Experiment}","page":"EEG","title":"BCIInterface.EEG.clear!","text":"clear!(experiment::Experiment)\n\nDelete all saved raw data from experiment.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.create_data-Tuple{String, BCIInterface.EEG.RawDataDescriptor}","page":"EEG","title":"BCIInterface.EEG.create_data","text":"create_data(name::String, data_desc::RawDataDescriptor)\n\nCreate new Data-Object for raw data (MCP3208).\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.create_data-Tuple{String, Device}","page":"EEG","title":"BCIInterface.EEG.create_data","text":"create_data(name::String, device::Device)\n\nCreate Data-Object which fits given device (raw data, fft data, etc.).\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.createpath-Tuple{String}","page":"EEG","title":"BCIInterface.EEG.createpath","text":"create_path(path::String)\n\nCreate necessary folders and file if they don't exist yet, so that isdir(path) or isfile(path) returns true depending on whether path points to a folder or a file.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.gather_data!-Tuple{Experiment, Number}","page":"EEG","title":"BCIInterface.EEG.gather_data!","text":"gather_data!(experiment::Experiment, runtime::Number; tags::Array=[], \nextra_info::Dict=Dict())\n\nGather raw EEG data. \n\nruntime: Runtime in seconds.\n\nOptional arguments:\n\ntags: Tags to add to every data point on top of tags given to Experiment\n\nextra_info: Extra info to add to every data point on top of extra info given  to Experiment\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.get_sample!-Tuple{BCIInterface.EEG.EEGBoard}","page":"EEG","title":"BCIInterface.EEG.get_sample!","text":"get_sample!(board::EEGBoard)\n\nUpdates board.sample to new data from board.\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.iscompatible-Tuple{DataFrames.DataFrame, Union{Nothing, BCIInterface.EEG.Metadata}}","page":"EEG","title":"BCIInterface.EEG.iscompatible","text":"is_compat(df::DataFrame, metadata::Union{Metadata, Nothing})\n\nCheck if df is \"compatible\" with metadata.\n\nIf metadata is nothing, then return true (for easy use with load_metadata which returns nothing when no metadata is defined)\n\n\n\n\n\n","category":"method"},{"location":"eeg/#BCIInterface.EEG.process-Tuple{Device, BCIInterface.EEG.StandardProcessor}","page":"EEG","title":"BCIInterface.EEG.process","text":"TODO: Processing\n\n1. Artifacts Removal\n2. Data Filtering\n3. Feature Extraction (FFT)\n\nAlso see this, page 2529.\n\n\n\n\n\n","category":"method"},{"location":"advanced_customisation/#Using-your-own-EEG-device","page":"Advanced Customisation","title":"Using your own EEG device","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BCIInterface","category":"page"},{"location":"#BCIInterface","page":"Home","title":"BCIInterface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BCIInterface.","category":"page"},{"location":"bci/#Creating,-managing-and-using-AI","page":"BCI","title":"Creating, managing and using AI","text":"","category":"section"},{"location":"bci/","page":"BCI","title":"BCI","text":"Modules = [BCIInterface]","category":"page"},{"location":"bci/","page":"BCI","title":"BCI","text":"Modules = [BCIInterface]","category":"page"},{"location":"developers/#Package-Development","page":"Package Development","title":"Package Development","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"Pages = [\"developers.md\"]\nDepth = 3","category":"page"},{"location":"developers/#Example-workflow-ideas","page":"Package Development","title":"Example workflow ideas","text":"","category":"section"},{"location":"developers/#Gathering-data","page":"Package Development","title":"Gathering data","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\ndevice = Device(MCP3208(\"/dev/spidev0.0\", 8))\nexperiment = Experiment(device, \"Test\", tags = [\"test\", \"significant\"], extra_info = Dict(:delay => 2), path = \"mydata/\")\nstates = [:left, :middle, :right]\nwhile true\n    for state in states\n        # Make testperson think of the $state side\n        sleep(2)\n        gather_data(device, \"data/test\", Seconds(10), tags = [state])\n    end\nend\nsave_data(experiment)","category":"page"},{"location":"developers/#Processing-data","page":"Package Development","title":"Processing data","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"data = load_data(\"Test\", :raw, path = \"mydata/\")\ndata_handler = DataHandler(\"Standard\", StandardFFT())\nprocessed_data = process_all(data, data_handler)\nsave_data(processed_data)","category":"page"},{"location":"developers/#Training-on-data","page":"Package Development","title":"Training on data","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\ndata = load_data(\"/data/test\")\nai = create_model(StandardOne(), data)\nai.max_accuracy = 0.9\ntrain!(ai, 100)","category":"page"},{"location":"developers/#Filtering-data","page":"Package Development","title":"Filtering data","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\nfunction myfilter(extra_info::Dict)::Bool\n    if haskey(extra_info, :delay) && extra_info[:delay] < 3\n        return true\n    end\n    return false\nend\ndata_filter = DataFilter(\n    include_tags = [[\"test\"], [\"significant\"]], \n    exclude_tags = [\"insignificant\"], \n    extra_info_filter = myfilter\n)\n\ndata = load_data(\"/data/test\", filter = data_filter)","category":"page"},{"location":"developers/#Creating-custom-models","page":"Package Development","title":"Creating custom models","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\ndata = load_data(\"/data/test\")\noutputs = [\n    (:left => [1.0, 1.0, 0.0, 0.0]),\n    (:none => [1.0, 0.0, 1.0, 0.0]),\n    (:right => [1.0, 0.0, 0.0, 1.0]),\n] # This seems very sensible...\nset_outputs!(data, outputs)\n\n\"\"\"\nMy own custom model. It's surely big enough to do any task!\n\"\"\"\nfunction my_own_model(input_shape, output_shape)\n    return @autosize (input_shape...) Chain(\n        Conv((3, 1), _ => 64),\n        flatten,\n        Dense(_, 100),\n        Dense(100, output_shape)\n    )\nend\n\nai = create_model(DefaultOne(), data, struct_constructer = my_own_model)\nai.max_accuracy = 0.9\ntrain!(ai, 100)","category":"page"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"See https://fluxml.ai/Flux.jl/stable/outputsize/.","category":"page"},{"location":"developers/#For-Package-Developers","page":"Package Development","title":"For Package Developers","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"This section is for everybody who wants to directly contribute to this package (and for us to not forget details!).","category":"page"},{"location":"developers/#Documentation","page":"Package Development","title":"Documentation","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"To preview the documentation locally before pushing to GitHub, use previewDocs.sh (Linux) or manually execute","category":"page"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"`julia --project=docs -ie 'using BCIInterface, LiveServer; servedocs()'`","category":"page"}]
}
