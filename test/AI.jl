using BCIInterface

outputs = Dict(
    :left => [1.0, 0.0, 0.0], :none => [0.0, 1.0, 0.0], :right => [0.0, 0.0, 1.0]
)

data = load_data("MCP3208"; dir=TEST_DIR)
modeldata = ModelData(data, outputs)
model = create_model(modeldata; modelname=MODEL_NAME, savedir=MODEL_DIR)

inputs = modeldata.dataloader.data.inputs
sample = selectdim(inputs, ndims(inputs), 1:4)
result = model(sample)

# check output dims (3 outputs, 4 samples)
@test size(result) == (3, 4) # TODO: get currect result for comparison
# check type stability
@test eltype(sample) == Float32
@test all(.==(typeof.(result), Float32))

save(model)
tmp_model = load_model(joinpath(MODEL_DIR, MODEL_NAME, ""))
@test model == tmp_model
