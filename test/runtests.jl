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

TEST_DIR = "testdata/"
TEST_NAME = "Test"
if isdir(TEST_DIR)
    rm(TEST_DIR, recursive=true)
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
        NUM_CHANNELS = 8
        NUM_COLS_META = 3 # time, tags, extraInfo
        EXTRA_INFO = Dict(:delay => 2)
        TAGS = ["test", "significant"]
        
        device = Device(MCP3208("/test/offline", NUM_CHANNELS, max_speed_hz=200, online=false))
        experiment = Experiment(device, TEST_NAME, tags=TAGS,
            extra_info=EXTRA_INFO, path=TEST_DIR)

        # set up necessary files etc. to avoid problems with timing
        save_data(experiment)
        # Compiling
        gather_data!(experiment, 0.1)
        # Actual execution
        gather_data!(experiment, 0.1)
        # With additional tags
        gather_data!(experiment, 0.1, tags = [:left, :foo])

        # Test if `getproperty` works
        @test experiment.num_samples == size(experiment.raw_data.df)[1]
        # Test if it added some elements
        @test experiment.num_samples > 10
        # Test if number of cols is correct
        @test size(experiment.raw_data.df)[2] == NUM_CHANNELS + NUM_COLS_META

        # save number of samples (in seperate var because of later clearing)
        total_num = experiment.num_samples

        # clear all data of experiment
        BCIInterface.EEG.clear!(experiment)
        @test experiment.num_samples == 0
        @test size(experiment.raw_data.df)[1] == 0

        # without saving in real-time
        gather_data!(experiment, 0.1, save = false)
        @test experiment.num_samples == size(experiment.raw_data.df)[1]
        @test experiment.num_samples > 1000 # Test if it added some elements

        total_num += experiment.num_samples
        # now save the unsaved data
        save_data(experiment)

        
        # load saved data into `data`
        data = load_data(TEST_DIR * TEST_NAME, "RawData", exact_num=true)
        # check if metadata is correct
        @test data.num_samples == size(data.df, 1)

        # direct, manual load
        df = BCIInterface.EEG.CSV.read("testdata/Test/RawData.csv", BCIInterface.EEG.DataFrame; skipto = 2)
        @test df == data.df

        # create new, empty experiment
        experiment2 = Experiment(device, TEST_NAME, tags=["test", "significant"],
            extra_info=Dict(:delay => 2), path=TEST_DIR)
        # load saved data into it
        load_data!(experiment2, exact_num=true)
        # check if metadata is correct
        @test experiment2.raw_data.num_samples == size(experiment2.raw_data.df, 1)

        # compare data of `data` with data of `experiment2`
        @test data.df == experiment2.raw_data.df
        # check if number of collected samples equals number of loaded samples
        @test total_num == data.num_samples
        @test total_num == experiment2.raw_data.num_samples

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
