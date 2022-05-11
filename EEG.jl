module EEG

using BrainFlow, PyPlot, FFTW

function get_some_board_data(board_shim, nsamples)
    data = BrainFlow.get_current_board_data(nsamples, board_shim)# |> transpose
    return data
    #eeg_data = data[:, 2:5]
    #=
    for chan = 1:4
        eeg_channel_data = view(eeg_data, :, chan)
        BrainFlow.detrend(eeg_channel_data, BrainFlow.CONSTANT)
    end
    =#
    #return eeg_data
end

function read_data_and_trans(num_of_files, location, color; delay=0.001)
    #fig, ax1 = subplot()
    #ax1.set_ylim(ymin = 0, ymax = 10000, auto = false)
    for i = 1:num_of_files
        data = BrainFlow.read_file(location * string(i) * ".csv")
        data = transpose(data)
        data = data[:, 2:5]
        for chan = 1:4
            eeg_channel_data = view(data, :, chan)
            BrainFlow.detrend(eeg_channel_data, BrainFlow.CONSTANT)
            plot(abs.(rfft(eeg_channel_data)), color)
        end
        #data = abs.(rfft(data[:, 1:4]))
        sleep(delay)
    end
end

function read_data(num_of_files, location, color)
    #fig, ax1 = subplot()
    #ax1.set_ylim(ymin = 0, ymax = 10000, auto = false)
    for i = 1:num_of_files
        data = BrainFlow.read_file(location * string(i) * ".csv")
        data = abs.(rfft(data[:, 1:4]))
        plot(data, color)
        sleep(0.0001)
    end
end

function get_file_index(path, index)
    if isfile(path * string(index) * ".csv")
        get_file_index(path, index + 1)
    else
        return index
    end
end

function get_eeg_train_data(board_shim)
    data = Array{Float64,2}
    blink_path = "Blink/Okzipital-03-16-2022/"
    no_blink_path = "NoBlink/Okzipital-03-16-2022/"

    next_i_blink = get_file_index(blink_path, 1)
    next_i_no_blink = get_file_index(no_blink_path, 1)


    for i = 0:9
        println("No Blink")
        print("\a")
        sleep(2)
        for i2 = 0:20
            sleep(0.25)
            data = get_some_board_data(board_shim, 200)
            BrainFlow.write_file(data, no_blink_path * string(next_i_no_blink + i * 20 + i2) * ".csv", "w")
            #figure("No Blink")
            #clf() 
            #plot(data)
        end

        println("Blink")
        sleep(2)
        for i2 = 0:20
            sleep(0.25)
            data = get_some_board_data(board_shim, 200)
            # (abs.(rfft(
            #print(size(data))
            #data[50] = 0
            #r_data = irfft(data, )
            BrainFlow.write_file(data, blink_path * string(next_i_blink + i * 20 + i2) * ".csv", "w")
            #figure("Blink")
            #clf()
            #plot(abs.(data))
        end
    end
end

function get_eeg_test_data(board_shim, blink_path, no_blink_path)

    blink_data = []
    no_blink_data = []

    its = 4

    for i = 1:its
        print("\a")
        println("Close eyes!")
        sleep(2.5)
        for i2 = 1:3
            sleep(2)
            push!(blink_data, get_some_board_data(board_shim, 400))
        end

        print("\a")
        println("Open eyes!")
        sleep(2.5)
        for i2 = 1:3
            sleep(2)
            push!(no_blink_data, get_some_board_data(board_shim, 400))
        end
    end

    next_i_blink = get_file_index(blink_path, 1)
    next_i_no_blink = get_file_index(no_blink_path, 1)

    for i = 1:length(blink_data)
        BrainFlow.write_file(blink_data[i], blink_path * string(next_i_blink + i - 1) * ".csv", "w")
    end

    for i = 1:length(no_blink_data)
        BrainFlow.write_file(no_blink_data[i], no_blink_path * string(next_i_no_blink + i - 1) * ".csv", "w")
    end

    BrainFlow.release_session(board_shim)
end

function setup_board(port_address) # nothing for synthetic
    BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
    if port_address === nothing
        params = BrainFlowInputParams() # Synthetic board
        board_shim = BrainFlow.BoardShim(BrainFlow.SYNTHETIC_BOARD, params)
    else
        params = BrainFlowInputParams(
            serial_port=port_address
        )
        board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
    end
    if BrainFlow.is_prepared(board_shim)
        BrainFlow.release_session(board_shim)
    end
    BrainFlow.prepare_session(board_shim)
    BrainFlow.start_stream(board_shim)
    return board_shim
end

function setup_board(os::Symbol) # :WIN for Windows, :LIN for Linux, :SYN for synthetic board
    if os == :WIN
        return setup_board("COM4")
    elseif os == :LIN
        return setup_board("/dev/cu.usbmodem11")
    elseif os == :SYN
        return setup_board(nothing)
    end
end

#board_shim = setup_board(:WIN)
blink_path = "Blink/livetest_data/Okzipital-05-11-2022/"
no_blink_path = "NoBlink/livetest_data/Okzipital-05-11-2022/"
one_blink_path = "OneBlink/Okzipital-05-11-2022/"

#get_eeg_test_data(board_shim, blink_path, one_blink_path)

next_i_blink = get_file_index(blink_path, 1)
next_i_no_blink = get_file_index(no_blink_path, 1)
next_i_one_blink = get_file_index(one_blink_path, 1)

read_data_and_trans(next_i_blink - 1, blink_path, "g", delay=0.1)
sleep(1)
read_data_and_trans(next_i_no_blink - 1, no_blink_path, "r", delay=0.1)
sleep(1)
read_data_and_trans(next_i_one_blink - 1, one_blink_path, "y", delay=0.1)



# get_eeg_train_data(board_shim)

#=
println("Blinks:")
read_data(201, "Blink/Okzipital-03-16-2022/", "g")
println("NoBlinks:")
read_data(201, "NoBlink/Okzipital-03-16-2022/", "r")
=#


end