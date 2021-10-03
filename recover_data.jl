module Recover


#println("Loading BrainFlow...")
using BrainFlow
#println("Loading BSON...")
using BSON
#println("Packages loaded!")


function get_endings_path(path)
    cd(path)
    i = 1
    endings = Float64[]
    while isfile(string(i) * ".csv")
        io = open(string(i) * ".csv")
        content = readlines(io)
        close(io)
        content = content[4] * "\t"

        str_i = 0
        for i = 1:199
            str_i += 1
            while collect(content[str_i])[1] != '\t'
                str_i += 1
            end
        end

        str_i = 1
        num_str = ""
        while collect(content[str_i])[1] != '\t'
            num_str *= content[str_i]
            str_i += 1
        end
        push!(endings, parse(Float64, num_str))
        i += 1
    end
    cd("..")
    return endings
end

function get_endings()
    endings = Array{Array{Float64,1},1}()
    push!(endings, get_endings_path("Blink"))
    push!(endings, get_endings_path("NoBlink"))
    return endings
end

end #Module