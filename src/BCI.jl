module BCI
export new_model, train!, get_data

# loading data
include("load_data.jl")

# creating & managing model
include("model.jl")

end