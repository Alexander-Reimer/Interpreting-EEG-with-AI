module BCIInterface

export func, sayhello

"""
    func(x)

Returns double the number `x` plus `1`.
"""
func(x) = 2x + 1

"""
    sayhello(name::String)

Give `name` a nice greeting!
"""
function sayhello(name::String)
    println("Hello, $name!")
end
# Write your package code here.

end
