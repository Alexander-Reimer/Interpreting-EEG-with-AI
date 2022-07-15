[comment]: <> "LTeX: language=en-US"
# Interpreting EEG with AI
## What is this?
This is a project by Alexander Reimer and Matteo Friedrich.

We are trying to create an open-source interface for recognizing ERPs of the human brain, for usage in brain–computer interfaces (BCIs). It is being made with people in mind who don't have a lot of experience or knowledge in the fields of EEG and AI.

We hope to give these people a chance to learn about these things and wake some interest, and make lives easier for people needing to use this technology.

We're just at the beginnings ourselves, however, and also want to learn and grow through this project. Any help and advice is appreciated!

This project is being developed with the German STEM competition Jugend forscht in mind.

### Goals

- Provide best possible results with any EEG, no matter how cheap
- Easy to use
    - provide good default options
    - uncomplicated adjustments to important parameters

## How do I use this?
!!! Outdated !!!
Please wait a bit until we get to updating this

We are still in the development stages, so for now, this isn't at all user friendly or well documented or commented as we mostly communicate in person or online and it's only the two of us working on the code.

Of course, if you are interested in this project or have questions, you can feel free to ask us (just check our profiles for contact information).

And if you for some reason decided to look through the code / theory and have suggestions or have found mistakes, feel free to tell us.

### If you want to execute what we already have,

- clone this repository,
- start Julia (make sure you're in the repo folder),
- open the package terminal by pressing `]`,
- type in `activate .`,
- type in `instantiate`,
- exit the package terminal by pressing backspace,
- type in `include("main.jl")` (note: the first start will take a long time).

To change parameters etc. you have to change the code itself for now (see **How does this work?** for further details).

## How does this work?
We are using the programming language Julia for this project.

For the neural network we are using the library [Flux](https://github.com/FluxML/Flux.jl). The network structure so far only consists of dense layers, the activation function we are using is the 𝜎-function. The structure itself isn't yet determined, we are always changing things like the amount of layers and the amount of neurons in each one (you can find the most recent one by looking in main.jl → function `build_model`).

The same goes for the training parameters like batchsize, learning rate, and whether the training data is shuffled after each iteration. You can find those in main.jl → mutable struct `Args`.

Before, the inputs of the neural network were just all voltages of all channels of the last second (4 channels, each 200hz → 4 * 200 = 800 inputs) so that the network has enough data but can also react quickly enough to make real-time control sensible.

But now, we first perform a dicrete Fourier transformation on the 200 samples of each channel and give the output of that to the network. We also cut off all frequencies above an upper limit or below a lower limit. These two values can also be found in main.jl → mutable struct `Args`. So in the end, the index of an input neuron corresponds to a certain frequency and the value is the amplitude of it. This means that the amount of input neurons depends on which frequencies you choose (lower and upper limit). For the Fourier transform, we used the package [FFTW](https://github.com/JuliaMath/FFTW.jl).

To gather the training data, we coded our own little program which you can find at EEG.jl. The EEG device we use is the Ganglion board by OpenBCI with 4 electrodes, and 2 earclips (1 for grounding, 1 for reference). The library we are using for reading the data of the EEG and saving it is [BrainFlow](https://brainflow.org).

This data is saved in Blink/ and NoBlink/.

The neural network has two possible output configurations, controlled by one_out. If `one_out` is `true`, then there is one output neuron, with 1.0 being a prediction of Blink and 0.0 bein a prediction of NoBlink. If `one_out` is `false`, the network has one output neuron for Blink and one for NoBlink. To determine the networks decision, we check which neuron has the higher value.

### Other packages

Other packages we use are

- [PyPlot](https://github.com/JuliaPy/PyPlot.jl) for plotting the cost development,
- [BSON](https://github.com/JuliaIO/BSON.jl) for saving and loading the network weights and cost history, and
- [CUDA](https://github.com/JuliaGPU/CUDA.jl) for utilising a Nvidia GPU if available.
- [Interpolations](https://github.com/JuliaMath/Interpolations.jl) for smoothing out the curves in some plots
