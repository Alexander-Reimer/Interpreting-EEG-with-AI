mutable struct Friend
    name::String
end

mutable struct Name
    name::String
end

struct Greeter
    greeting::String
    name::Name
    friend::Friend
end

function Greeter(greeting::String, name::String)
    return Greeter(greeting, Name(name), Friend(""))
end

"""
    greet(greeter::Greeter)
Greet person defined in Greeter.
"""
function greet(g::Greeter)
    println(g.greeting, " ", g.name.name, "!")
end

function change_name(g::Greeter, new_name::String)
    g.name.name = new_name
end

function add_friend(g::Greeter, friend_name::String)
    g.friend.name = friend_name
end

g = Greeter("Hello", "Alex")
greet(g)
change_name(g, "Tom")
greet(g)
add_friend(g, "Tim")