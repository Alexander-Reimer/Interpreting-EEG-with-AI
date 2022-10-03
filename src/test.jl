function myprint3(arg)
    println(typeof(arg))
    println(arg)    
end

function myprint3(args...)
    for arg in args
        myprint3(arg)
    end
end