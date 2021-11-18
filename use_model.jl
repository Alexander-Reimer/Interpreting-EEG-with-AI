module use_model
println("Loading BrainFlow...")
using BrainFlow

println("Loading Flux..")
#using Flux

#println("Loading CUDA...")
#using CUDA

function get_EEG_data(board_shim, time)
    
end
BrainFlow.enable_dev_logger(BrainFlow.BOARD_CONTROLLER)
params = BrainFlowInputParams(
    serial_port = "/dev/ttyACM0"
)
board_shim = BrainFlow.BoardShim(BrainFlow.GANGLION_BOARD, params)
BrainFlow.prepare_session(board_shim)
BrainFlow.start_stream(board_shim)


BrainFlow.stop_stream(board_shim)
BrainFlow.release_session(board_shim)

end # Module