using BCIInterface
using Test

@testset "BCIInterface.jl" begin
    @testset "EEG" begin
        # Creation of MCP3208 (offline mode)
        @test typeof(MCP3208("/test/offline", 8, max_speed_hz = 200, online = false)) <: MCP3208
        @test typeof(Device(MCP3208("/test/offline", 8, max_speed_hz = 200, online = false))) <: Device
        dev = Device(MCP3208("/test/offline", 8, max_speed_hz = 200, online = false))
        @test typeof(BCIInterface.EEG.get_voltage(dev.board, 1)) <: Number
        @test_throws "Given channel" BCIInterface.EEG.get_voltage(dev.board, 9)
        @test_throws "Given channel" BCIInterface.EEG.get_voltage(dev.board, 0)
    end
    # Write your tests here.
end
