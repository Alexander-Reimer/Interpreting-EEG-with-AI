# Supported Boards

## OpenBCI GUI

You can use the [OpenBCI GUI](https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/) to get EEG data.

You need to do the following steps in the GUI:
1. Create a stream in the GUI by connecting your OpenBCI board, streaming from a file, using a synthetic stream, ...
2. Open the "Networking" widget.
3. Change the protocol to "LSL" in the top right corner
4. Set the data type in Stream 1 to "FFT"
5. Enable "Filter" if you want
6. Click on "Start LSL Stream
7. Click on "Start Data Stream" in the top left

Now, you can create the `Board` object responsible for receiving the data in Julia using
```julia
board = GanglionGUI(NUM_CHANNELS)
```
where `NUM_CHANNELS` exactly equals the number of channels whatever data source connected to the GUI has.

## MCP3208

TODO