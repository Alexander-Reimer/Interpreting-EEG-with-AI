ev3dev_path = "../ev3dev.jl/ev3dev.jl"
include(ev3dev_path)
setup("../mount/sys/class")
include("main.jl")

left_motor = Motor(:outB)
right_motor = Motor(:outC)

robot = Robot(left_motor, right_motor)

drive(robot, 0)

for speed = 1:100
    drive(robot, speed)
    sleep(0.05)
end