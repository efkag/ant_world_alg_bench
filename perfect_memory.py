from utils import rotate
import correlation_coefficient
import idf

class PerfectMemory:

    def __init__(self, route_images, matching, degree_shift=1):
        self.route_images = route_images
        self.degree_shift = degree_shift
        self.degrees = list(range(0, 360, degree_shift))
        self.degrees_iter = range(0, 360, degree_shift)
        self.recovered_heading = []
        self.logs = []
        if matching == 'corr':
            self.matcher = correlation_coefficient.CorrelationCoefficient()
        elif matching == 'idf':
            self.matcher = idf.RotationalIDF()
        else:
            raise Exception('Non valid matching method name')

    def navigate(self, w_g_imgs):
        for i in range(0, len(w_g_imgs)):  # For every world grid image
            current_image = w_g_imgs[i]  # set the current image

            mem_sims = []  # Memory similarity between the current image and route images
            mem_headings = []  # Hold the recovered Headings for the current image by the different route images
            route_logs = []
            for j in range(0, len(self.route_images)):  # For every goal Image

                goal_image = self.route_images[j]  # Set the goal Image
                rsims = []  # Hold the rsims between the current and the image of the route for every degree
                for k in self.degrees_iter:
                    # Rotate the current image
                    curr_image = rotate(k, current_image)
                    # IDF function to find the error between the selected route image and the rotated current
                    rsims.append(self.matcher.match(curr_image, goal_image))

                # log the rsims for that on route image on all the degrees
                route_logs.append(rsims)
                best, index = self.matcher.best_match(rsims)
                mem_sims.append(best)
                mem_headings.append(self.degrees[index])

            self.logs.append(route_logs)  # append the rsims of all route images for that wg image
            best, index = self.matcher.best_match(mem_sims)
            self.recovered_heading.append(mem_headings[index])

        return self.recovered_heading, self.logs
