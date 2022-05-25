include("main.jl")
using BrainFlow

ev3dev_path = "../ev3dev.jl/ev3dev.jl"
include(ev3dev_path)
setup()

#left_middle = Motor(:outC)
#right_middle = Motor(:outA)
left_motor = Motor(:outB)
right_motor = Motor(:outD)

robot = Robot(left_motor, right_motor)

function setdown_own()
    deactivate(robot)
end

drive(robot, 0)

function robot_test(modelo)

    BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    params = BrainFlowInputParams(
        serial_port="COM3"
    )
    board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)

    try
        BrainFlow.release_session(board_shim)
    catch
        nothing
    end
    samples = []
    #BrainFlow.release_session(board_shim)
    BrainFlow.prepare_session(board_shim)
    BrainFlow.start_stream(board_shim)
    sleep(1)
    println("Starting!")
    println("")
    blink_vals = []
    no_blink_vals = []
    for i = 1:40
        sample = EEG.get_some_board_data(board_shim, 200)
        #clf()
        #plot(sample)
        sample = [make_fft(sample[1:200])..., make_fft(sample[201:400])...,
            make_fft(sample[401:600])..., make_fft(sample[601:800])...]

        y = model(sample)

        push!(blink_vals, y[1])
        push!(no_blink_vals, y[2])

        if y[1] > 0.5
            drive(robot, 70)
        else
            drive(robot, 0)
        end

        clf()
        plot(blink_vals, "green")
        plot(no_blink_vals, "red")

        sleep(0.25)
    end
    BrainFlow.release_session(board_shim)
    drive(robot, 0)
end

#train(true)


global hyper_params = Args(0.001, 0, "model.bson", cuda=false, one_out=true, plot_frequency=100, fft=true, lower_limit=1, upper_limit=20, batch_size=1)
global model = build_model()
load_network!("model.bson")

robot_test(model)