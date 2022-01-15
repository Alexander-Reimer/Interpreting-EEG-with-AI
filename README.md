# Interpreting EEG with AI
## What is this?
This is a project by Alexander Reimer and Matteo Friedrich.

We are trying to teach a neural network to interpret event-related potentials of the human brain. For this we use EEG data which we are gathering with our 4 channel Ganglion board from OpenBCI.

Currently, we are at the beginning so we are just training the network to recognize blinks for now. But the endgoal is for a human to control a roboter real-time using this.

We want to participate in the German STEM-competition Jugend Forscht 2022 with it.

## How do I use this?
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

To gather the training data, we coded our own little program which you can find at EEG.jl. The EEG device we use is the Ganglion board by OpenBCI with 2 flat electrodes, 2 spike electrodes, and 2 earclips (1 for grounding, 1 for reference). The library we are using for reading the data of the EEG and saving it is [BrainFlow](https://brainflow.org).

This data is saved in Blink/ and NoBlink/.

The neural network has 2 output neurons: The first one for Blink and the second one for Not a blink. To determine the networks decision, we check which neuron has the higher value.

### Other packages

Other packages we use are

- [PyPlot](https://github.com/JuliaPy/PyPlot.jl) for plotting the cost development,
- [BSON](https://github.com/JuliaIO/BSON.jl) for saving and loading the network weights and cost history, and
- [CUDA](https://github.com/JuliaGPU/CUDA.jl) for utilising the GPU if available.
