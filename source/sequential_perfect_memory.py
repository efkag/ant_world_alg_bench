from source import correlation_coefficient, idf
from source.utils import rotate
import numpy as np


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
        self.confidence = [1] * self.route_end
        self.window_sims = []
        self.CMA = []
        if matching == 'corr':
            self.matcher = correlation_coefficient.CorrelationCoefficient()
            self.prev_match = 1.0
        elif matching == 'rmse':
            self.matcher = idf.RotationalIDF()
            self.prev_match = 255
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
            window_logs = []

            for j in range(mem_pointer, limit):  # For every goal Image
                goal_image = self.route_images[j]  # Set the goal Image
                # similarities between the current and every rotated route image
                rsims = []  # Rotational similarities

                for k in self.degrees_iter:  # For every degree
                    curr_image = rotate(k, current_image)  # Rotate the current image
                    # IDF function to find the error between the selected route image and the rotated current
                    rsims.append(self.matcher.match(curr_image, goal_image))
                # log the rsims between the current image and the jth route image
                window_logs.append(rsims)

                # get best similarity match for degrees and its index(heading)
                best, index = self.matcher.best_match(rsims)
                wind_sims.append(best)
                wind_headings.append(self.degrees[index])

            # Save the best degree match for window similarities
            self.window_sims.append(wind_sims)
            # append the rsims of all window route images for that current image
            self.logs.append(window_logs)
            best, index = self.matcher.best_match(wind_sims)
            self.recovered_heading.append(wind_headings[index])
            # recovered_heading.append(sum(wind_headings)/len(wind_headings))
            # Dynamic window adaptation based on match gradient.
            # if best > self.prev_match or window < 5:
            #     window += 1
            # elif window > 20:
            #     window -= 2
            # else:
            #     window -= 1
            # self.prev_match = best
            #
            # # Lower confidence of the memories depending on the match score
            # window_mean = sum(wind_sims)/len(wind_sims)
            # if i == 0: # If this is the first window
            #     self.CMA.extend([window_mean] * 2)
            # else:
            #     cma = self.CMA[-1]
            #     self.CMA.append(cma + ((window_mean-cma)/(len(self.CMA)+1)))
            # for j in range(mem_pointer, limit):
            #     if wind_sims[j-mem_pointer] > self.CMA[-1]:
            #         self.confidence[j] -= 0.1

            # Update memory pointer
            mem_pointer = index + mem_pointer
            self.matched_index_log.append(mem_pointer)

        return self.recovered_heading, self.logs, self.window_log

    def get_index_log(self):
        return self.matched_index_log

    def get_confidence(self):
        return self.confidence

    def get_window_sims(self):
        return self.window_sims

    def get_CMA(self):
        return self.CMA


class Seq2SeqPerfectMemory:

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
        self.confidence = [1] * self.route_end
        self.window_sims = []
        self.CMA = []
        if matching == 'corr':
            self.matcher = correlation_coefficient.CorrelationCoefficient()
            self.prev_match = 1.0
        elif matching == 'rmse':
            self.matcher = idf.RotationalIDF()
            self.prev_match = 255
        else:
            raise Exception('Non valid matching method name')

    def navigate(self, w_g_imgs, window=10, mem_pointer=0):
        no_of_test_imgs = len(w_g_imgs)
        end_route = False
        current_img_pointer = -1
        # For every sequence in the grid images
        while not end_route:

            # Update current images pointer
            current_img_pointer += 1
            current_limit = current_img_pointer + window
            if current_limit >= no_of_test_imgs:
                current_limit = no_of_test_imgs
                end_route = True
            # Get the current image sequence
            current_imgs = w_g_imgs[current_img_pointer:current_limit]

            # Window similarities between the current image and the window route images
            wind_sims = []
            # Recovered headings for the current image wrt the window
            wind_headings = []
            # Update memory window pointers
            limit = mem_pointer + window
            if limit > self.route_end: limit = self.route_end
            self.window_log.append([mem_pointer, limit])
            window_sims = []

            # Compare current images to memory window images
            i = 0
            for j in range(mem_pointer, limit):  # For every goal Image
                # Set the goal_image and the current image
                goal_image = self.route_images[j]
                current_img = current_imgs[i]
                # similarities between the current and every rotated route image
                rsims = []  # Rotational similarities

                for k in self.degrees_iter:  # For every degree
                    rot_img = rotate(k, current_img)  # Rotate the current image
                    # IDF function to find the error between the selected route image and the rotated current
                    rsims.append(self.matcher.match(rot_img, goal_image))
                # log the rsims between the current image and the jth route image
                window_sims.append(rsims)

                # get best similarity match for degrees and its index(heading)
                best, index = self.matcher.best_match(rsims)
                wind_sims.append(best)
                wind_headings.append(self.degrees[index])

                i += 1

            # Save the best degree match for window similarities
            self.window_sims.append(wind_sims)
            # append the rsims of all window route images for that current image
            self.logs.append(window_sims)
            best, index = self.matcher.best_match(wind_sims)
            self.recovered_heading.append(wind_headings[index])
            # recovered_heading.append(sum(wind_headings)/len(wind_headings))
            # Dynamic window adaptation based on match gradient.
            # if best > self.prev_match or window < 5:
            #     window += 1
            # elif window > 20:
            #     window -= 2
            # else:
            #     window -= 1
            # self.prev_match = best

            # # Lower confidence of the memories depending on the match score
            # window_mean = sum(wind_sims)/len(wind_sims)
            # if i == 0: # If this is the first window
            #     self.CMA.extend([window_mean] * 2)
            # else:
            #     cma = self.CMA[-1]
            #     self.CMA.append(cma + ((window_mean-cma)/(len(self.CMA)+1)))
            # for j in range(mem_pointer, limit):
            #     if wind_sims[j-mem_pointer] > self.CMA[-1]:
            #         self.confidence[j] -= 0.1

            # Update memory pointer
            mem_pointer = index + mem_pointer
            self.matched_index_log.append(mem_pointer)

        return self.recovered_heading, self.logs, self.window_log

    def get_index_log(self):
        return self.matched_index_log

    def get_confidence(self):
        return self.confidence

    def get_window_sims(self):
        return self.window_sims

    def get_CMA(self):
        return self.CMA