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

function test_experiment(board::BCIInterface.EEG.EEGBoard, name::String, gather_time)
    device = Device(board)
    experiment = Experiment(device, name, tags=["test", "significant"],
        extra_info=Dict(:delay => 2), path=TEST_DIR)

    # set up necessary files etc. to avoid problems with timing
    save_data(experiment)
    # Compiling
    gather_data!(experiment, 0.1)
    # Actual execution
    gather_data!(experiment, 0.1)
    # With additional tags
    gather_data!(experiment, 0.1, tags=[:left, :foo])
    # Test if it added some elements
    @test experiment.num_samples > 10

    # Test if `getproperty` works
    @test experiment.num_samples == size(experiment.data.df)[1]

    # save number of samples (in seperate var because of later clearing)
    total_num = experiment.num_samples

    # clear all data of experiment
    BCIInterface.EEG.clear!(experiment)
    @test experiment.num_samples == 0
    @test size(experiment.data.df)[1] == 0

    # without saving in real-time
    gather_data!(experiment, gather_time, save=false)
    @test experiment.num_samples == size(experiment.data.df)[1]
    # Test if it added some elements
    @test experiment.num_samples > 100

    total_num += experiment.num_samples

    # now save the unsaved data
    save_data(experiment)
    # load saved data into `data`
    data = load_data(TEST_DIR * name, "RawData", exact_num=true)
    # check if metadata is correct
    @test data.num_samples == size(data.df, 1)

    # direct, manual loading
    df = BCIInterface.EEG.CSV.read("testdata/" * name * "/RawData.csv", BCIInterface.EEG.DataFrame; skipto=2)

    # create new, empty experiment
    experiment2 = Experiment(device, name, tags=["test", "significant"],
        extra_info=Dict(:delay => 2), path=TEST_DIR)
    # load saved data into it
    load_data!(experiment2, exact_num=true)
    # check if metadata is correct
    @test experiment2.data.num_samples == size(experiment2.data.df, 1)

    # compare data of `data` with data of `experiment2`
    @test df == data.df
    @test data.df == experiment2.data.df
    # check if number of collected samples equals number of loaded samples
    @test total_num == data.num_samples
    @test total_num == experiment2.data.num_samples
end

TEST_DIR = "testdata/"
if isdir(TEST_DIR)
    rm(TEST_DIR, recursive=true)
end

@testset "BCIInterface.jl" begin
    @testset "Experiment" begin
        NUM_CHANNELS = 8
        NUM_COLS_META = 3 # time, tags, extraInfo

        @testset "MCP3208" begin
            board = MCP3208("/test/offline", NUM_CHANNELS, max_speed_hz=200,
                online=false)

            dev = Device(board)
            @test is(BCIInterface.EEG.get_voltage(dev.board, 1), Number)
            # Test if channel limits are checked
            @test_throws "Given channel" BCIInterface.EEG.get_voltage(dev.board, 9)
            @test_throws "Given channel" BCIInterface.EEG.get_voltage(dev.board, 0)
            
            test_experiment(board, "MCP3208", 0.005)
        end
        # TODO: implement for CI
        @testset "GanglionGUI" begin
            board = GanglionGUI(NUM_CHANNELS)
            test_experiment(board, "GanglionGUI", 0.01)
        end
    end
    # @testset "Data Saving & Loading"
    @testset "Data processing" begin
        function creation_tests()
            _ = BCIInterface.EEG.Standard()
            return true
        end
        @test creation_tests()

        std_processor = BCIInterface.EEG.Standard()
        # @test BCIInterface.process()
    end
end