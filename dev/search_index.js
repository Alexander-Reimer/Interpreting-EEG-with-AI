var documenterSearchIndex = {"docs":
[{"location":"eeg/#Gathering,-processing-and-loading-EEG-data","page":"EEG","title":"Gathering, processing and loading EEG data","text":"","category":"section"},{"location":"eeg/","page":"EEG","title":"EEG","text":"Modules = [BCIInterface.EEG]","category":"page"},{"location":"eeg/","page":"EEG","title":"EEG","text":"Modules = [BCIInterface.EEG]","category":"page"},{"location":"eeg/#BCIInterface.EEG.Channel","page":"EEG","title":"BCIInterface.EEG.Channel","text":"Single Channel with voltages in Volt and the times the voltages were recorded at in seconds.\n\n\n\n\n\n","category":"type"},{"location":"eeg/#BCIInterface.EEG.Device","page":"EEG","title":"BCIInterface.EEG.Device","text":"Device for gathering EEG data. Create it using\n\nDevice(board::EEGBoard)\n\n... TODO\n\n\n\n\n\n","category":"type"},{"location":"eeg/#BCIInterface.EEG.get_voltages-Tuple{Device}","page":"EEG","title":"BCIInterface.EEG.get_voltages","text":"Get current voltages of device.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = BCIInterface","category":"page"},{"location":"#BCIInterface","page":"Home","title":"BCIInterface","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for BCIInterface.","category":"page"},{"location":"bci/#Creating,-managing-and-using-AI","page":"BCI","title":"Creating, managing and using AI","text":"","category":"section"},{"location":"bci/","page":"BCI","title":"BCI","text":"Modules = [BCIInterface]","category":"page"},{"location":"bci/","page":"BCI","title":"BCI","text":"Modules = [BCIInterface]","category":"page"},{"location":"bci/#BCIInterface.func-Tuple{Any}","page":"BCI","title":"BCIInterface.func","text":"func(x)\n\nReturns double the number x plus 1.\n\n\n\n\n\n","category":"method"},{"location":"bci/#BCIInterface.sayhello-Tuple{String}","page":"BCI","title":"BCIInterface.sayhello","text":"sayhello(name::String)\n\nGive name a nice greeting!\n\n\n\n\n\n","category":"method"},{"location":"developers/#Package-Development","page":"Package Development","title":"Package Development","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"Pages = [\"developers.md\"]\nDepth = 3","category":"page"},{"location":"developers/#Example-workflow-ideas","page":"Package Development","title":"Example workflow ideas","text":"","category":"section"},{"location":"developers/#Gathering-data","page":"Package Development","title":"Gathering data","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\ndevice = Device(MCP3208(\"/dev/spidev0.0\", 8))\ndata_handler = DataHandler(StandardFFT(), [:left, :middle, :right], \"data/test\", 200)\nwhile true\n    for state in [:left, :middle, :right]\n        # Make testperson think of the $state side\n        gather_data(device, data_handler, :right, Seconds(10))\n    end\nend","category":"page"},{"location":"developers/#Training-on-data","page":"Package Development","title":"Training on data","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\ndata = load_data(\"/data/test\")\nai = create_model(StandardOne(), data)\nai.max_accuracy = 0.9\ntrain!(ai, 100)","category":"page"},{"location":"developers/#Creating-custom-models","page":"Package Development","title":"Creating custom models","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"using BCIInterface\n\ndata = load_data(\"/data/test\")\noutputs = [\n    (:left => [1.0, 1.0, 0.0, 0.0]),\n    (:none => [1.0, 0.0, 1.0, 0.0]),\n    (:right => [1.0, 0.0, 0.0, 1.0]),\n] # This seems very sensible...\nset_outputs!(data, outputs)\n\n\"\"\"\nMy own custom model. It's surely big enough to do any task!\n\"\"\"\nfunction my_own_model(input_shape, output_shape)\n    return @autosize (input_shape...) Chain(\n        Conv((3, 1), _ => 64),\n        flatten,\n        Dense(_, 100),\n        Dense(100, output_shape)\n    )\nend\n\nai = create_model(DefaultOne(), data, struct_constructer = my_own_model)\nai.max_accuracy = 0.9\ntrain!(ai, 100)","category":"page"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"See https://fluxml.ai/Flux.jl/stable/outputsize/.","category":"page"},{"location":"developers/#For-Package-Developers","page":"Package Development","title":"For Package Developers","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"This section is for everybody who wants to directly contribute to this package (and for us to not forget details!).","category":"page"},{"location":"developers/#Documentation","page":"Package Development","title":"Documentation","text":"","category":"section"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"To preview the documentation locally before pushing to GitHub, use previewDocs.sh (Linux) or manually execute","category":"page"},{"location":"developers/","page":"Package Development","title":"Package Development","text":"`julia --project=docs -ie 'using BCIInterface, LiveServer; servedocs()'`","category":"page"}]
}
