using BCIInterface
using Test

@testset "BCIInterface.jl" begin
    function test1()
        if func(5) != 11
            throw(AssertionError("Doesn't work!!"))
        end
    end
    test1()
    # Write your tests here.
end
