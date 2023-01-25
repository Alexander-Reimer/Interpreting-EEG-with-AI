# Package Development
```@contents
Pages = ["developers.md"]
Depth = 3
```

This is the first package we have ever developed. This means that it will hopefully improve significantly over time, but also that any help, experiences and advice would be greatly appreciated.

Feel free to contact us over email ([alexander.reimer2357@gmail.com](mailto:alexander.reimer2357@gmail.com)) or open an issue on the GitHub repository.

## Design Principle

This package has two main goals: Firstly to make the experience for the user as simple, fast and uncomplicated as possible, and secondly make it useable in every scenario.

To combine these two, we decided to make this package as modular and adjustable as possible.
This way, we can provide "default" modules, so people don't have to deal with every single aspect, while still making it possible to adjust anything if a project requires it.

For example, say you use an unsupported EEG or want to use a different EEG processing pipeline -- this way, you won't need to recode big chunks of the program for small changes.

## Example workflow ideas
### Gathering data
```julia
using BCIInterface

device = Device(MCP3208("/dev/spidev0.0", 8))
experiment = Experiment(device, "Test", tags = ["test", "significant"], extra_info = Dict(:delay => 2), path = "mydata/")
states = [:left, :middle, :right]
while true
    for state in states
        # Make testperson think of the $state side
        sleep(2)
        gather_data(device, "data/test", Seconds(10), tags = [state])
    end
end
save_data(experiment)
```
### Processing data
```julia
data = load_data("Test", :raw, path = "mydata/")
data_handler = DataHandler("Standard", StandardFFT())
processed_data = process_all(data, data_handler)
save_data(processed_data)
```
### Training on data
```julia
using BCIInterface

data = load_data("/data/test")
ai = create_model(StandardOne(), data)
ai.max_accuracy = 0.9
train!(ai, 100)
```
### Filtering data
```julia
using BCIInterface

function myfilter(extra_info::Dict)::Bool
    if haskey(extra_info, :delay) && extra_info[:delay] < 3
        return true
    end
    return false
end
data_filter = DataFilter(
    include_tags = [["test"], ["significant"]], 
    exclude_tags = ["insignificant"], 
    extra_info_filter = myfilter
)

data = load_data("/data/test", filter = data_filter)
```
#### Creating custom models
```julia
using BCIInterface

data = load_data("/data/test")
outputs = [
    (:left => [1.0, 1.0, 0.0, 0.0]),
    (:none => [1.0, 0.0, 1.0, 0.0]),
    (:right => [1.0, 0.0, 0.0, 1.0]),
] # This seems very sensible...
set_outputs!(data, outputs)

"""
My own custom model. It's surely big enough to do any task!
"""
function my_own_model(input_shape, output_shape)
    return @autosize (input_shape...) Chain(
        Conv((3, 1), _ => 64),
        flatten,
        Dense(_, 100),
        Dense(100, output_shape)
    )
end

ai = create_model(DefaultOne(), data, struct_constructer = my_own_model)
ai.max_accuracy = 0.9
train!(ai, 100)
```
See [https://fluxml.ai/Flux.jl/stable/outputsize/](https://fluxml.ai/Flux.jl/stable/outputsize/).

## Documentation
To preview the documentation locally before pushing to GitHub, use `previewDocs.sh` (Linux) or manually execute

    `julia --project=docs -ie 'using BCIInterface, LiveServer; servedocs()'`
