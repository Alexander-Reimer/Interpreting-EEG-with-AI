using BCIInterface

board = MCP3208("", 8, online = false)
dev = Device(board)
data = gather_data(dev)