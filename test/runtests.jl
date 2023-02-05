using BCIInterface
using Test

const TEST_DIR = "tmpdata/"
const MODEL_DIR = "tmpmodels/"
const MODEL_NAME = "TestModel"

is(var, t::DataType) = typeof(var) <: t ? true : false

for dir in [TEST_DIR, MODEL_DIR]
    if isdir(dir)
        rm(dir; recursive=true)
    end
end

@testset "BCIInterface.jl" begin
    include("EEG.jl")
    include("AI.jl")
end
