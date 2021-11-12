import numpy as np
import matplotlib.pyplot as plt
import cv2

EXPERIMENT_AREA_X = (-3000.0, 3000.0)
EXPERIMENT_AREA_Y = (-3000.0, 3000.0)

EXPERIMENT_AREA_WIDTH = EXPERIMENT_AREA_X[1] - EXPERIMENT_AREA_X[0]
EXPERIMENT_AREA_HEIGHT = EXPERIMENT_AREA_Y[1] - EXPERIMENT_AREA_Y[0]
RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)

data = np.loadtxt("ground-pos 1_Trajectories_100.csv", delimiter=",", skiprows=5, dtype=np.float32)
coords = data[:,2:]
avg_coords = np.average(coords, axis=0)
std_coords = np.std(coords, axis=0)
assert np.all(std_coords < 0.1)

# Reshape into columns, throw away z and take transpose
avg_coords = np.reshape(avg_coords, (-1, 3))
avg_coords = avg_coords[:,:2]
avg_coords = np.copy(avg_coords)

# Check there's enough points
assert avg_coords.shape[0] > 3

# Because image coordinates can't be negative, offset everything to edge of experiment area
avg_coords[:,0] -= EXPERIMENT_AREA_X[0]
avg_coords[:,1] -= EXPERIMENT_AREA_Y[0]

# Read image and convert to RGB
image = cv2.imread("115_0002.JPG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

fig, axis = plt.subplots()

# Setup array of point colours
current_point = 0
point_colours = [RED for i in range(avg_coords.shape[0])]
point_colours[0] = GREEN
scatter_actors = axis.scatter(avg_coords[:,0], avg_coords[:,1], c=point_colours)

image_points = np.zeros(avg_coords.shape, dtype=np.float32)

image_actor = axis.imshow(image)#, extent=(min_x, max_x - min_x, min_y, max_y - min_y))
print(image_actor.get_extent())
axis.set_xlim((0, EXPERIMENT_AREA_WIDTH))
axis.set_ylim((0, EXPERIMENT_AREA_HEIGHT))


def on_click(event):
    global current_point
    
    if image_actor.contains(event):
        display_to_axis_transform = axis.transData.inverted()
        
        # Transform event coordinate to 
        coordinate = display_to_axis_transform.transform((event.x, event.y))
        
        # Record image point corresponding to current point
        image_points[current_point,:] = coordinate

        # If we've got enough points for affine
        if current_point == 2:
            # Calculate affine transform and apply
            affine_transform = cv2.getAffineTransform(image_points[:3,:], avg_coords[:3,:])
            warp_image = cv2.warpAffine(image, affine_transform, (int(EXPERIMENT_AREA_WIDTH),
                                                                  int(EXPERIMENT_AREA_HEIGHT)))
            # Save warp image
            cv2.imwrite("warped.png", cv2.cvtColor(warp_image,cv2.COLOR_RGB2BGR))
            
            # Display
            image_actor.set_data(warp_image)
            image_actor.set_extent((-0.5, EXPERIMENT_AREA_WIDTH - 0.5, EXPERIMENT_AREA_HEIGHT - 0.5, -0.5))
        
        point_colours[current_point] = RED
        current_point = (current_point + 1) % image_points.shape[0]
        point_colours[current_point] = GREEN
        scatter_actors.set_color(point_colours)
        
        fig.canvas.draw()

def on_key(event):
    global avg_coords, current_point, image_points
    print(event.key)
    if event.key == "x":
        image_points = np.delete(image_points, current_point, 0)    
        avg_coords = np.delete(avg_coords, current_point, 0)   
        scatter_actors.set_offsets(avg_coords)
        fig.canvas.draw()
    
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
