module AI

using PyCall, Flux, PyPlot
np = pyimport("numpy")
using Flux: crossentropy, train!
using BSON: @save, @load

include("default_config.jl") # provide default options, don't change
include("config.jl") # overwrite default options, you just need to set the variables

function get_data(path)
    return np.load(path)
end

function get_all_data(folder)
    # Get data of all files in $folder
    # Return 3D-Array, (sample, channel, amplitude of frequency given by fft)
    files = readdir(folder)
    l = 0
    for file in files
        d = get_data(folder * file)
        l += size(d)[1]
    end

    data = zeros(Float32, l, 16, 60)
    i = 0
    index = 1

    for file in files
        #println(i)
        d = get_data(folder * file)
        i += 1
        data[index:index+size(d)[1]-1, :, :] = d
        index += size(d)[1]
    end
    return data
end

function get_formatted_data(path, output, test)
    # Transform data, add outputs
    # Add data X and Y to global variables X_traindata, X_testdata, Y_traindata, Y_testdata
    data = get_all_data(path)
    data = mapslices(rotr90, data, dims=[1, 3])

    if test
        global X_testdata[:, 1, :, index+1:index+size(data)[3]] = data
    else
        global X_traindata[:, 1, :, index+1:index+size(data)[3]] = data
    end

    for i = 1:size(data)[3]
        if test
            global Y_testdata[:, index+i] = output
        else
            global Y_traindata[:, index+i] = output
        end
    end
    global index += size(data)[3]
end

function check_NaN(model)
    # See if gradients contain NaNs
    ps = Flux.params(model)
    gs = Flux.gradient(ps) do
        loss(X_traindata, Y_traindata)
    end
    search_NaN = []
    for elements in gs
        push!(search_NaN, 1 ∈ isnan.(elements))
    end
    return search_NaN
end

function get_accuracy(model, data_x, data_y)
    total = 0
    correct = 0
    all_out = model(data_x)
    for i = 1:size(data_x)[4]
        est = all_out[:, i]
        if argmax(est) == argmax(data_y[:, i])
            correct += 1
        end
        total += 1
    end
    return correct / total
end

# *************************************************************************************************************************
# DATA
num_outputs = length(TRAIN_DATA[1][2])

# Pre-initialise arrays to improve performance
global X_traindata = Array{Float32}(undef, MAX_FREQUENCY, 1, NUM_CHANNELS, NUM_SAMPLES_TRAIN)
global Y_traindata = Array{Float32}(undef, num_outputs, NUM_SAMPLES_TRAIN)

global X_testdata = Array{Float32}(undef, MAX_FREQUENCY, 1, NUM_CHANNELS, NUM_SAMPLES_TEST)
global Y_testdata = Array{Float32}(undef, num_outputs, NUM_SAMPLES_TEST)

# Populate arrays with eeg data
global index = 0
for (path, output) in TRAIN_DATA
    get_formatted_data(path, output, false)
end

global index = 0
for (path, output) in TEST_DATA
    get_formatted_data(path, output, true)
end

# Turn data into Flux DataLoaders
global train_data = Flux.Data.DataLoader((X_traindata, Y_traindata), batchsize=BATCH_SIZE, shuffle=true, partial=false)
global test_data = Flux.Data.DataLoader((X_testdata, Y_testdata), batchsize=BATCH_SIZE, shuffle=true, partial=false)

# Clear unnecessary data
# X_traindata = nothing
# Y_traindata = nothing
# X_testdata = nothing
# Y_testdata = nothing

# *************************************************************************************************************************
# TRAINING

model = MODEL()

#=
model = Chain(
   Conv((5,1), 16=>64, relu),
   Conv((5,1), 64=>128, relu),
   Conv((5,1), 128=>256, relu),
   Conv((5,1), 256=>512, relu),
   Conv((16,1), 512=>3, relu),
   Flux.flatten,
   Dense(87, 3),
   softmax
)
=#


loss(x, y) = LOSS(model(x), y)
ps = Flux.params(model)
opt = OPTIMIZER(LEARNING_RATE)

train_loss_history = []
test_loss_history = []

train_loss = loss(X_traindata, Y_traindata)
test_loss = loss(X_testdata, Y_testdata)

push!(train_loss_history, train_loss)
push!(test_loss_history, test_loss)
println(0, " train: ", train_loss)
println(0, " test: ", test_loss)
println(get_accuracy(model, X_testdata, Y_testdata))


for i = 1:1
    #println(check_NaN(model))
    for (x, y) in train_data
        gs = Flux.gradient(() -> Flux.Losses.mse(model(x), y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
        #println("calculadet batch")
    end

    #train!(loss, ps, train_data, opt)

    if i % 1 == 0
        train_loss = loss(X_traindata, Y_traindata)
        test_loss = loss(X_testdata, Y_testdata)
        push!(train_loss_history, train_loss)
        push!(test_loss_history, test_loss)
        println(i, " train: ", train_loss)
        println(i, " test: ", test_loss)
    end
end
@save "model.bson" model
plot(train_loss_history)
plot(test_loss_history)
println(model(X_traindata[:, :, :, 1:1]))

#=
data_ = [Array{Float32}(data_[i,:,:]) for i in 1:size(data_)[1]]
data = Vector{Matrix{Float32}}(undef, length(data_))
for i = 1:length(data_)
    data[i] = reshape(data_[i], 1, :)
end
println("transformed data", length(data))
=#

end #module