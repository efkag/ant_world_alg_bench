import antworld
import cv2

# Old Seville data (lower res, but loads faster)
# worldpath = antworld.bob_robotics_path + "/resources/antworld/world5000_gray.bin"
# z = 0.01 # m

# New Seville data
worldpath = antworld.bob_robotics_path + "/resources/antworld/seville_vegetation_downsampled.obj"
print(antworld.bob_robotics_path)
z = 1.5 # m (for some reason the ground is at ~1.5m for this world)

agent = antworld.Agent(720, 150)
(xlim, ylim, zlim) = agent.load_world(worldpath)
print(xlim, ylim, zlim)
xstart = xlim[0] + (xlim[1] - xlim[0]) / 2.0
y = ylim[0] + (ylim[1] - ylim[0]) / 2.0

print("starting at (%f, %f, %f)" % (xstart, y, z))

for x in range(3):
    agent.set_position(x + xstart, y, z)
    # agent.set_attitude(yaw, pitch, roll)
    im = agent.read_frame()
    filename = "antworld%i.png" % x
    print("Saving image as %s..." % filename)
    cv2.imwrite(filename, im)

