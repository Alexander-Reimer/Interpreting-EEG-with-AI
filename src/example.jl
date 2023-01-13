include("BCI.jl")
# using .BCI

include("config.jl") # defines variable "config"

data = BCI.get_data(config)
model = BCI.new_model(config)
# model = BCI.load_model("model-logging/allConv_4c_1_2022-12-10_18-26-25") # ≈ 45% auf ganglion
# model = BCI.load_model("model-logging/allConv_4c_1_2022-12-10_18-46-06/model_2022-12-10_18-46-47.bson")

# BCI.train!(model, data)