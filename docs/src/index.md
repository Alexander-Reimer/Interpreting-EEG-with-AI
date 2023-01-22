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

```julia
using BCIInterface
```

To start collecting EEG data, first create a `board` using the instructions in [Supported Boards](@ref).

Then create a `Device` and `Experiment` with it:

```julia
device = Device(board)
experiment = Experiment(device, "NameOfMyExperiment")
```

Now, say you want to later let an AI predict what colour a test person is thinking of.
For this, you want the AI to classify the data using three categories: The person thinking of red, blue and yellow.

A possible setup would be to first make the test person think of red for 5 seconds (by telling them to or showing it on a screen) and start gathering data at the same time using

```julia
gather_data!(experiment, 5, tags=[:red])
```

Then, you repeat the same for yellow and blue, replacing `[:red]` by `[:yellow]` and `[:blue]` respectively.

---

Now you can start to train an AI using your collected data.

First, you need to actually create the model using...

TODO