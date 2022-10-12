module LoadConfig
export load_config

function load_config(path, name, module_space)
    io = open(path)
    config_txt = readlines(io)
    close(io)
    cache_path = "_cache-config-$name.jl"
    io2 = open(cache_path, "w")
    truncate(io2, 0)
    write(io2, "module $name \n")
    for line in config_txt
        write(io2, line * "\n")
    end
    write(io2, "end")
    close(io2)
    include(cache_path)
    return 
end
end