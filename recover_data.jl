module Recover

#=
Whenever the EEG data is read out using the BrainFlow package, 
the very last signal of every training sample gets corrupted (different and seemingly random output every time).
We still haven't found the reason for it so for now, this is the temporary fix.
It opens every file, and goes through the underlying text data of the .csv to get the last number.
=#

using BrainFlow

function get_endings_path(path)
    pwd_cur = pwd()
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
    cd(pwd_cur)
    return endings
end

function get_endings(blink_path, no_blink_path)
    blink_endings = get_endings_path(blink_path)
    no_blink_endings = get_endings_path(no_blink_path)

    return (blink = blink_endings, no_blink = no_blink_endings)
end

end #Module