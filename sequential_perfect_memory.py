from utils import idf, rotate


def seq_perf_mem(w_g_imgs, route_images, window=10, mem_pointer=0):
    degree_shift = 1
    degrees = list(range(0, 360, degree_shift))
    route_end = len(route_images)
    recovered_heading = []
    logs = []
    window_log = []

    for i in range(0, len(w_g_imgs)):  # For every world grid image
        current_image = w_g_imgs[i]  # set the current image

        mem_ridfs = []  # Hold the RMS between the current image and print(len(w_g_imgs_inrange))the different route images
        mem_heading = []  # Hold the recovered Headings for the current image by the different route images
        limit = mem_pointer + window
        if limit > route_end: limit = route_end
        window_log.append([mem_pointer, limit])
        route_logs = []

        for j in range(mem_pointer, limit):  # For every goal Image
            goal_image = route_images[j]  # Set the goal Image
            ridfs = []  # Hold the ridf between the current and the image of the route for every degree

            for k in range(0, 360, degree_shift):
                curr_image = rotate(k, current_image)  # Rotate the current image
                # IDF function to find the error between the selected route image and the rotated current
                ridfs.append(idf(curr_image, goal_image))
            route_logs.append(ridfs)  # log the RMS between the current view and the jth route image on all the degrees
            mem_ridfs.append(min(ridfs))
            mem_heading.append(degrees[ridfs.index(min(ridfs))])

        logs.append(route_logs)  # append the rIDF (rRMSE) of all route images for that wg image
        recovered_heading.append(mem_heading[mem_ridfs.index(min(mem_ridfs))])
        # recovered_heading.append(sum(mem_heading)/len(mem_heading))
        mem_pointer = mem_ridfs.index(min(mem_ridfs)) + mem_pointer

    return recovered_heading, logs, window_log
