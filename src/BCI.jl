module BCI
export new_model, train!, get_datak

# loading data
include("load_data.jl")

# creating & managing model 
include("model.jl")
include("pruning.jl")

end