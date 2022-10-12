module AI2
[include(file) for file in ["load_data.jl", "model.jl", "config.jl"]]
using .DataLoader, .Model, .Config

data = get_data(Config.TRAIN_DATA, Config.TEST_DATA)

end