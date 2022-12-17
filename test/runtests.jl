using BCIInterface
using Test

function is(var, t)
    return typeof(var) <: t ? true : false
end

macro executes(expr1)
    return true
end

function test_lengths(lengths, expected_len)
    for len in lengths
        if len[1] != expected_len || len[2] != expected_len
            return false
        end
    end
    return true
end

@testset "BCIInterface.jl" begin
    @testset "EEG-Devices" begin
        function creation_tests()
            _ = MCP3208("/test/offline", 8, max_speed_hz=200, online=false)
            _ = Device(MCP3208("/test/offline", 8, max_speed_hz=200, online=false))
            return true
        end
        @test creation_tests()

        devices = [
            Device(MCP3208("/test/offline", 8, max_speed_hz=200, online=false)),
        ]
        for dev in devices
            @test is(BCIInterface.EEG.get_voltage(dev.board, 1), Number)
            # Test if channel limits are checked
            @test_throws "Given channel" BCIInterface.EEG.get_voltage(dev.board, 9)
            @test_throws "Given channel" BCIInterface.EEG.get_voltage(dev.board, 0)
            # Test if update_data adds exactly one voltage to every channel
            BCIInterface.EEG.update_data!(dev)
            lengths = map(x -> [length(x.voltages), length(x.times)], dev.channels)
            @test test_lengths(lengths, 1)
        end
    end
    @testset "Experiment" begin
        function creation_tests(device)
            _ = Experiment(device, "Test", tags=["test", "significant"],
                extra_info=Dict(:delay => 2), path="mydata/")
            return true
        end
        device = Device(MCP3208("/test/offline", 8, max_speed_hz=200, online=false))
        @test creation_tests(device)
        global experiment = Experiment(device, "Test", tags=["test", "significant"],
            extra_info=Dict(:delay => 2), path="testdata/")

        BCIInterface.EEG.clear!(experiment)
        # set up necessary files etc. to avoid problems with timing
        save_data(experiment)
        # Compiling
        gather_data!(experiment, 0.1, tags=[:left])
        # Actual execution
        gather_data!(experiment, 0.1, tags=[:left])
        @test experiment.num_samples == size(experiment.raw_data.df)[1]
        @test experiment.num_samples > 10 # Test if it added some elements
        @test size(experiment.raw_data.df)[2] == 3 + 8
        
        BCIInterface.EEG.clear!(experiment)
        gather_data!(experiment, 0.1, tags=[:left], save=false)
        @test experiment.num_samples == size(experiment.raw_data.df)[1]
        @test experiment.num_samples > 1000 # Test if it added some elements
        @test size(experiment.raw_data.df)[2] == 3 + 8
        save_data(experiment)
        # Known problem: compat check fails, because CSV.read of metadata
        # results in df.colmn_names = "[\"time\", \"tags\", \"extraInfo\", \"1\", \"2\", \"3\", \"4\", \"5\", \"6\", \"7\", \"8\"]"
        gather_data!(experiment, 0.1, tags=[:left])
    end
    # @testset "Data Saving & Loading"
    @testset "Data processing" begin
        function creation_tests()
            _ = BCIInterface.EEG.Standard()
            return true
        end
        @test creation_tests()

        std_processor = BCIInterface.EEG.Standard()
        # @test BCIInterface.EEG.process()
    end
end
