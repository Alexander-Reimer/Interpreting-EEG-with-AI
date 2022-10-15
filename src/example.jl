include("BCI.jl")
# using .BCI

include("config.jl")
conf = Config.init_config()
println(typeof(conf))

data = BCI.get_data(conf)
# model = BCI.load_model("model-logging/conv_sentdex_2022-10-08_11:43:28/")
model = BCI.new_model(conf)

BCI.train!(model, data)

# BCI.train!(model2, data)