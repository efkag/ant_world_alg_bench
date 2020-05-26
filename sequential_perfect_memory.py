import correlation_coefficient
import idf
from utils import rotate


class SequentialPerfectMemory:

    def __init__(self, route_images, matching, degree_shift=1):
        self.route_end = len(route_images)
        self.route_images = route_images
        self.degree_shift = degree_shift
        self.degrees = list(range(0, 360, degree_shift))
        self.degrees_iter = range(0, 360, degree_shift)
        self.recovered_heading = []
        self.logs = []
        self.window_log = []
        self.matched_index_log = []
        if matching == 'corr':
            self.matcher = correlation_coefficient.CorrelationCoefficient()
        elif matching == 'idf':
            self.matcher = idf.RotationalIDF()
        else:
            raise Exception('Non valid matching method name')

    def navigate(self, w_g_imgs, window=10, mem_pointer=0):
        for i in range(0, len(w_g_imgs)):  # For every world grid image
            current_image = w_g_imgs[i]  # set the current image

            # Window similarities between the current image and the window route images
            wind_sims = []
            wind_headings = []  # Recovered headings for the current image
            limit = mem_pointer + window
            if limit > self.route_end: limit = self.route_end
            self.window_log.append([mem_pointer, limit])
            route_logs = []

            for j in range(mem_pointer, limit):  # For every goal Image
                goal_image = self.route_images[j]  # Set the goal Image
                # similarities between the current and every rotated route image
                rsims = []  # Rotational similarities

                for k in self.degrees_iter:  # For every degree
                    curr_image = rotate(k, current_image)  # Rotate the current image
                    # IDF function to find the error between the selected route image and the rotated current
                    rsims.append(self.matcher.match(curr_image, goal_image))
                # log the rsims between the current image and the jth route image
                route_logs.append(rsims)

                # get best similarity match for degrees and its index(heading)
                best, index = self.matcher.best_match(rsims)
                wind_sims.append(best)
                wind_headings.append(self.degrees[index])

            # append the rsims of all window route images for that current image
            self.logs.append(route_logs)
            best, index = self.matcher.best_match(wind_sims)
            self.recovered_heading.append(wind_headings[index])
            # recovered_heading.append(sum(wind_headings)/len(wind_headings))
            # Update memory pointer
            mem_pointer = index + mem_pointer
            self.matched_index_log.append(mem_pointer)

        return self.recovered_heading, self.logs, self.window_log

    def get_index_log(self):
        return self.matched_index_log


class Seq2PerfectMemory:

    def __init__(self, route_images, matching, degree_shift=1):
        self.route_end = len(route_images)
        self.route_images = route_images
        self.degree_shift = degree_shift
        self.degrees = list(range(0, 360, degree_shift))
        self.degrees_iter = range(0, 360, degree_shift)
        self.recovered_heading = []
        self.logs = []
        self.window_log = []
        if matching == 'corr':
            self.matcher = correlation_coefficient.CorrelationCoefficient()
        elif matching == 'idf':
            self.matcher = idf.RotationalIDF()
        else:
            raise Exception('Non valid matching method name')