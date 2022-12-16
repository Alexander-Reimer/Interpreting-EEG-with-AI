# using BCIInterface

device = Device(MCP3208("/dev/spidev0.0", 8, online = false))
device = Device(MCP3208("/dev/spidev0.0", 8, online = false))
experiment = Experiment(device, "Test", tags=["test", "significant"],
    extra_info=Dict(:delay => 2), path="mydata/")
states = [:left, :middle, :right]
# while true
    for state in states
        # Make testperson think of the $state side
        gather_data!(experiment, 1, tags = [state], delay = 0.00)
    end
# end
# save_data(experiment)