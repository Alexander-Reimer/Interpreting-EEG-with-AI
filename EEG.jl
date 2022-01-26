module EEG

using BrainFlow, PyPlot

function get_some_board_data(board_shim, nsamples)
    data = BrainFlow.get_current_board_data(nsamples, board_shim) |> transpose
    eeg_data = data[:, 2:5]
    for chan = 1:4
        eeg_channel_data = view(eeg_data, :, chan)
        BrainFlow.detrend(eeg_channel_data, BrainFlow.CONSTANT)
    end
    return eeg_data
end

function read_data(num_of_files, location)
    for i = 1:num_of_files
        data = BrainFlow.read_file(location * string(i) * ".csv")
        data = data[:, 1:2]
        figure("Restored Data")
        clf()
        plot(data)
        sleep(0.1)
    end
end

function get_latest(path, index)
    if isfile(path * string(index) * ".csv")
        get_latest(path, index + 1)
    else
        return index
    end
end

function main(board_shim)
    data = Array{Float64,2}
    blink_path = "Blink/01-26-2022/"
    no_blink_path = "NoBlink/01-26-2022/"

    next_i_blink = get_latest(blink_path, 1)
    next_i_no_blink = get_latest(no_blink_path, 1)

    for i = 0:100
        println("No Blink")
        sleep(1)
        data = get_some_board_data(board_shim, 200)
        BrainFlow.write_file(data, no_blink_path * string(next_i_no_blink + i) * ".csv", "w")
        figure("No Blink")
        clf() 
        plot(data)

        println("Blink")
        sleep(1)
        data = get_some_board_data(board_shim, 200)
        BrainFlow.write_file(data, blink_path * string(next_i_blink + i) * ".csv", "w")
        figure("Blink")
        clf()
        plot(data)
    end
end




#=
BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)

# params = BrainFlowInputParams() # Synthetic board

#= params = BrainFlowInputParams(
    serial_port = "/dev/cu.usbmodem11"
)=# # Ganglion board Kubuntu

params = BrainFlowInputParams(
    serial_port = "COM3"
) # Ganglion board Windows
#board_shim = BrainFlow.BoardShim(BrainFlow.SYNTHETIC_BOARD, params)
board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)

BrainFlow.prepare_session(board_shim)
BrainFlow.start_stream(board_shim)
println(get_some_board_data(board_shim, 200))
#main(board_shim)
BrainFlow.release_session(board_shim)
=#

#=
println("Blinks:")
read_data(100, "Blink/")
println("NoBlinks:")
read_data(100, "NoBlink/")
=#
#=
BrainFlow.release_session(board_shim)
=#
#read_data(10, "Blink/")

end