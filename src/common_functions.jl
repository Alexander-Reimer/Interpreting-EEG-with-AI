function save_model()
    global model
    model = model |> cpu
    bson(SAVE_PATH,
        model=model,
        channels=NUM_CHANNELS,
        frequency=MAX_FREQUENCY,
        x_history=x_history,
        train_loss_history=train_loss_history,
        test_loss_history=test_loss_history,
        train_accuracy_history=train_accuracy_history,
        test_accuracy_history=test_accuracy_history)
end

function load_model()
    global NUM_CHANNELS
    global MAX_FREQUENCY
    data = BSON.load(LOAD_PATH, @__MODULE__)
    if data[:channels] != NUM_CHANNELS
        NUM_CHANNELS = data[:channels]
        @warn "Replaced NUM_CHANNELS in config by NUM_CHANNELS used for loaded model since they don't match."
    end
    if data[:frequency] != MAX_FREQUENCY
        MAX_FREQUENCY = data[:frequency]
        @warn "Replaced MAX_FREQUENCY in config by MAX_FREQUENCY used for loaded model since they don't match."
    end

    global x_history = data[:x_history]
    global train_loss_history = data[:train_loss_history]
    global test_loss_history = data[:test_loss_history]
    global train_accuracy_history = data[:train_accuracy_history]
    global test_accuracy_history = data[:test_accuracy_history]
    global model = data[:model]
end