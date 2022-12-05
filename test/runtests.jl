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
        function test_data_update()            
            BCIInterface.EEG.update_data!(dev)
            lengths = map(x -> [length(x.voltages), length(x.times)], dev.channels)
            for len in lengths
                if len[1] != 1 || len[2] != 1
                    return false
                end
            end
            return true
        end
        @test test_data_update()
        function test_creation_Standard()
            global std_processor = BCIInterface.EEG.Standard()
            return true
        end
        @test test_creation_Standard()
        # @test BCIInterface.EEG.process()
    end
end
