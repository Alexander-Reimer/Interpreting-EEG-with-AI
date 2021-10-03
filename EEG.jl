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
        data = BrainFlow.read_file(location*string(i)*".csv")
        figure("Restored Data")
        clf()
        plot(data)
        sleep(1)
    end
end

function main(board_shim)
    data = Array{Float64, 2}
    for i in 1:10
        sleep(1)
        data = get_some_board_data(board_shim, 200)
        BrainFlow.write_file(data, "NoBlink/"*string(i)*".csv", "w")
        figure("No Blink")
        clf()
        plot(data)

        sleep(1)
        data = get_some_board_data(board_shim, 200)
        BrainFlow.write_file(data, "Blink/"*string(i)*".csv", "w")
        figure("Blink")
        clf()
        plot(data)
    end
end

#=
BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
params = BrainFlowInputParams()
board_shim = BrainFlow.BoardShim(BrainFlow.SYNTHETIC_BOARD, params)
BrainFlow.prepare_session(board_shim)
BrainFlow.start_stream(board_shim)
=#
read_data(100, "Blink/")
read_data(100, "NoBlink/")
#=
BrainFlow.release_session(board_shim)
=#
#read_data(10, "Blink/")

end