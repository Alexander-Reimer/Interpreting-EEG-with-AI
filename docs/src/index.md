```@meta
CurrentModule = BCIInterface
```

# BCIInterface

Documentation for [BCIInterface](https://github.com/AR102/Interpreting-EEG-with-AI).

> **WARNING**: This package is currently undergoing large changes and refactoring, meaning
> the documentation may be out of date and the current development branch ("refactor") isn't
> finished yet.

## Installation

You can install this package with:

```julia
using Pkg
Pkg.add(url="https://github.com/AR102/Interpreting-EEG-with-AI", rev="refactor")
```

## Basic Usage

Include the package in your code with

```@example ex1
using BCIInterface
```

To start collecting EEG data, first create an [`EEGBoard`](@ref) using the instructions in
[Supported Boards](@ref).

An offline example for trying out the rest of the framework would be

```@example ex1
board = MCP3208("UselessBoard", 8, online = false)
nothing # hide
```

Then create an `Experiment`](@ref) with it:

```@example ex1
experiment = Experiment(board, "NameOfMyExperiment")
nothing # hide
```

Now, say you want to later let an AI predict what colour a test person is thinking of.
For this, you want the AI to classify the data using three categories: The person thinking
of red, blue and yellow.

A possible setup would be to first make the test person think of red for 5
seconds (by telling them to or showing it on a screen) and start gathering data
at the same time using [`gather_data!`](@ref)

```julia
gather_data!(experiment, 5, tags=[:red])
nothing # hide
```

```@example ex1
gather_data!(experiment, 0.3, tags=[:red], autosave = false) # hide
gather_data!(experiment, 0.3, tags=[:yellow], autosave = false) # hide
gather_data!(experiment, 0.3, tags=[:blue], autosave = false) # hide
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
```

Now you can create a [`Model`](@ref) (neural network) by using [`create_model`](@ref):

```@example ex1
model = create_model(modeldata)
```

To apply this model to some inputs:

```@example ex1
# take first 100 samples from modeldata
sample = modeldata.dataloader.data.inputs[:, :, 1:100]
# calculate results, should be 3x100 (3 outputs, 100 samples)
model(sample)
```

To train the model, use [`train!`](@ref):

```julia
train!(model, modeldata, epochs=5)
```