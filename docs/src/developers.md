# Package Development
```@contents
Pages = ["developers.md"]
Depth = 3
```
## Example workflow ideas
### Gathering data
```julia
using BCIInterface

device = Device(MCP3208("/dev/spidev0.0", 8))
data_handler = DataHandler(StandardFFT(), [:left, :middle, :right], "data/test", 200)
while true
    for state in [:left, :middle, :right]
        # Make testperson think of the $state side
        gather_data(device, data_handler, :right, Seconds(10))
    end
end
```
### Training on data
```julia
using BCIInterface

data = load_data("/data/test")
ai = create_model(StandardOne(), data)
ai.max_accuracy = 0.9
train!(ai, 100)
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
## For Package Developers
This section is for everybody who wants to directly contribute to this package (and for us to not forget details!).
### Documentation
To preview the documentation locally before pushing to GitHub, use `previewDocs.sh` (Linux) or manually execute

    `julia --project=docs -ie 'using BCIInterface, LiveServer; servedocs()'`
