# using https://github.com/tlienart/LiveServer.jl/#serve-docs
julia --project=docs -ie 'using BCIInterface, LiveServer; servedocs()'