using BCIInterface
NUM_CHANNELS = 8

mcp_board = MCP3208("", NUM_CHANNELS; online=false)

@test BCIInterface.get_voltage(mcp_board, 1) == 0
# Test if channel limits are checked
@test_throws "Given channel" BCIInterface.get_voltage(mcp_board, 9)
@test_throws "Given channel" BCIInterface.get_voltage(mcp_board, 0)

boards = [mcp_board]
board_names = ["MCP3208"]

for (board, board_name) in zip(boards, board_names)
    BCIInterface.get_sample!(board)
    @test board.sample == [0 for i in 1:NUM_CHANNELS]

    experiment = Experiment(board, board_name; tags=["test", "test2"], dir=TEST_DIR)

    gather_data!(experiment, 0.3)
    @test experiment.data.df[1, :tags] == ["test", "test2"]
    gather_data!(experiment, 0.3; tags=["left"])
    gather_data!(experiment, 0.3; tags=["none"])
    gather_data!(experiment, 0.3; tags=["right"])
    gather_data!(experiment, 0.3; tags=["nonrelevant"])
    @test experiment.data.df[end, :tags] == ["test", "test2", "nonrelevant"]

    # Check if data is duplicated when loading already saved data
    tmp_data = copy(experiment.data)
    load_data!(experiment)
    @test tmp_data == experiment.data

    # Check if data is loaded correctly
    tmp_data = load_data(experiment.data.name; dir=TEST_DIR)
    # because extraInfo (Dict) isn't parsed correctly
    @test_broken tmp_data == experiment.data
    # check if tags was parsed correctly
    @test tmp_data.df[end, :tags] == ["test", "test2", "nonrelevant"]
end
