ev3dev_path = "../ev3dev.jl/ev3dev.jl"
include(ev3dev_path)
setup("Z:/Programming/EEG/mount/sys/class/")
include("main.jl")

#left_middle = Motor(:outC)
#right_middle = Motor(:outA)
left_motor = Motor(:outB)
right_motor = Motor(:outD)

robot = Robot(left_motor, right_motor)

function setdown_own()
    deactivate(robot)
end

drive(robot, 0)

for speed = 1:100
    drive(robot, speed)
    sleep(0.05)
end