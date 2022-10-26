include("BCI.jl")
# using .BCI

include("config.jl") # defines variable "config"

data = BCI.get_data(config)
# model = BCI.load_model("model-logging/conv_sentdex_2022-10-08_11:43:28/")
model = BCI.new_model(config)