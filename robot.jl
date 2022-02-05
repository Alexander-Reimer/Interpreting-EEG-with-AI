ev3dev_path = "../ev3dev.jl/ev3dev.jl"
include(ev3dev_path)
setup("Z:/Programming/EEG/mount/sys/class/")
include("main.jl")

#left_middle = Motor(:outC)
#right_middle = Motor(:outA)
left_motor = Motor(:outB)
right_motor = Motor(:outD)

robot = Robot(left_motor, right_motor)

function setdown_own()
    deactivate(robot)
end

drive(robot, 0)

function test2(model)
    AI.BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    params = AI.BrainFlowInputParams(
        serial_port = "COM3"
    )
    board_shim = AI.BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
    samples = []
    #BrainFlow.release_session(board_shim)
    AI.BrainFlow.prepare_session(board_shim)
    AI.BrainFlow.start_stream(board_shim)
    sleep(1)
    println("Starting!")
    println("")
    blink_vals = []
    no_blink_vals = []
    for i = 1:100
        sample = AI.EEG.get_some_board_data(board_shim, 200)
        sample = [make_fft(sample[1:200])..., make_fft(sample[201:400])...]
    
        y = AI.model(sample)
    
        push!(blink_vals, y[1])
        push!(no_blink_vals, y[2])
    
        if y[2] > y[1]# + 0.08
            drive(robot, 70)
        else
            drive(robot, 0)
        end
    
        clf()
        plot(blink_vals, "green")
        plot(no_blink_vals, "red")

        #=
        sample = reshape(sample, (:, 1))
        println(y[1], "    ", y[2])
        push!(blink_vals, y[1])
        push!(no_blink_vals, y[2])
    
        #clf()
        plot(blink_vals, "green")
        plot(no_blink_vals, "red")
        if y[1] > y[2] + 0.2
            #println("hgizugz")
        end
        #push!(samples, sample)
        =#
        sleep(0.25)
        #print("\b\b\b\b\b")
        #println(counter)
    end
    AI.BrainFlow.release_session(board_shim)
end
#=
device = prepare_cuda()
model = build_model()
parameters = old_network()
Flux.loadparams!(model, parameters)
test2(model)
=#