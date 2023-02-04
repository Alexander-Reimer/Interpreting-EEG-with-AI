```@meta
CurrentModule = BCIInterface
```

# BCIInterface

Documentation for [BCIInterface](https://github.com/AR102/Interpreting-EEG-with-AI).

> **WARNING**: This package is currently undergoing large changes and refactoring, meaning the documentation may be out of date and the current development branch ("refactor") isn't finished yet.

## Installation

> Note: Not working right now!
You can install this package with:

```julia
using Pkg
Pkg.add("https://github.com/AR102/Interpreting-EEG-with-AI#refactor")
```

## Basic Usage

Include the package in your code with

```@example ex1
using BCIInterface
```

To start collecting EEG data, first create an [`EEGBoard`](@ref) using the instructions in [Supported Boards](@ref).

An offline example for trying out the rest of the framework would be

```@example ex1
board = MCP3208("UselessBoard", 8, online = false)
nothing # hide
```

Then create a [`Device`](@ref) and [`Experiment`](@ref) with it:

```@example ex1
device = Device(board)
experiment = Experiment(device, "NameOfMyExperiment")
nothing # hide
```

Now, say you want to later let an AI predict what colour a test person is thinking of.
For this, you want the AI to classify the data using three categories: The
person thinking of red, blue and yellow.

A possible setup would be to first make the test person think of red for 5
seconds (by telling them to or showing it on a screen) and start gathering data
at the same time using [`gather_data!`](@ref)

```julia
gather_data!(experiment, 5, tags=[:red])
```

```@example ex1
gather_data!(experiment, 0.3, tags=[:red], save = false) # hide
gather_data!(experiment, 0.3, tags=[:yellow], save = false) # hide
gather_data!(experiment, 0.3, tags=[:blue], save = false) # hide
```

Then, you repeat the same for yellow and blue, replacing `[:red]` by `[:yellow]`
and `[:blue]` respectively.

To access the data, you can either use the [`Experiment`](@ref) object if you're
still in the same session

```@example ex1
data = experiment.data
```

Or you can load it from the file where it should have been automatically saved:
```julia
data = load_data("NameOfMyExperiment", "RawData")
```

---

Now you can start to train an AI using your collected data.

First, you need to create a [`ModelData`](@ref) object.
This stores the data in a format usable by an artificial neural network.

In this format, every category of inputs that we want to classify is assigned an
output which the model should return whenever it recognizes this category.

This is done via a `Dict` where each key is the label of all inputs that belong
into one category and its value the output.

In our case, this could be something like

```@example ex1
outputs = Dict(
    :red => [1.0, 0.0, 0.0],
    :yellow => [0.0, 1.0, 0.0],
    :blue => [0.0, 0.0, 1.0]
)
```

And now to create the [`ModelData`](@ref) object:

```@example ex1
modeldata = ModelData(data, outputs)
println(size(modeldata.dataloader.data.inputs))
```

Now, you can create a `Model` (neural network) by using [`create_model`](@ref):

```@example ex1
model = create_model(modeldata)
```

To apply this model to some inputs:

```@example ex1
# take first 100 samples from modeldata
sample = modeldata.dataloader.data.inputs[:, :, 1:100]
model(sample)
```