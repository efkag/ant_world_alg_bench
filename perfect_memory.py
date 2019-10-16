from utils import idf, rotate


def perf_mem(w_g_imgs, on_route_images):
    degree_shift = 1
    IDF_error = []
    Recovered_Heading = []
    logs = []

    for i in range(0, len(w_g_imgs)):  # For every world grid image
        current_image = w_g_imgs[i]  # set the current image

        RMS_For_Goal_Image = []  # Hold the RMS between the current image and print(len(world_grid_imgs10))the different route images
        Headings_For_Goal_image = []  # Hold the recovered Headings for the current image by the different route images
        route_logs = []
        for j in range(0, len(on_route_images)):  # For every goal Image

            goal_image = on_route_images[j]  # Set the goal Image
            RMS = []  # Hold the RMS between the current and the image of the route for every degree
            Degrees = []  # Hold the degrees

            for k in range(0, 360, degree_shift):
                # Rotate the current image
                curr_image = rotate(k, current_image)
                # IDF function to find the error between the selected route image and the rotated current
                RMS.append(idf(curr_image, goal_image))
                Degrees.append(k)  # Degrees

            # log the RMS for that on route image on all the degrees
            route_logs.append(RMS)
            RMS_For_Goal_Image.append(min(RMS))
            Headings_For_Goal_image.append(Degrees[RMS.index(min(RMS))])

        logs.append(route_logs)  # append the RMS of all route images for that wg image
        IDF_error.append(min(RMS_For_Goal_Image))
        Recovered_Heading.append(Headings_For_Goal_image[RMS_For_Goal_Image.index(min(RMS_For_Goal_Image))])

    return IDF_error, Recovered_Heading, logs