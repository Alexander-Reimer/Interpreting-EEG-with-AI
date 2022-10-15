include("BCI.jl")
# using .BCI

include("config.jl")
conf = Config

data = BCI.get_data(conf)
# model = BCI.load_model("model-logging/conv_sentdex_2022-10-08_11:43:28/")
new_model = BCI.new_model(conf)
sleep(2)
trained_model = BCI.new_model(conf)

BCI.train!(trained_model, data)

# BCI.train!(model2, data)