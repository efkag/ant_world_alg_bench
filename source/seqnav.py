from source.utils import mae, rmse, cor_dist, rmf, pair_rmf
import numpy as np


class SequentialPerfectMemory:

    def __init__(self, route_images, matching, deg_range=(0, 360), degree_shift=1, window=20, adaptive=False):
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.deg_step = degree_shift
        self.degrees = np.arange(*deg_range)

        # Log Variables
        self.recovered_heading = []
        self.logs = []
        self.window_log = []
        self.matched_index_log = []
        self.confidence = [1] * self.route_end
        self.window_sims = []
        self.CMA = []
        #
        matchers = {'corr': cor_dist, 'rmse': rmse, 'mae': mae}
        self.matcher = matchers.get(matching)
        if not self.matcher:
            raise Exception('Non valid matcher method name')
        self.argminmax = np.argmin
        self.prev_match = 0.0

        # Window parameters
        self.mem_pointer = 0
        assert window > 2
        self.window = window
        self.adaptive = adaptive

    def get_heading(self, query_img, dynamic_window=False):
        # TODO:Need to update this function to keep the memory pointer (best match)
        # TODO: in the middle of the window
        # get the rotational similarities between a query image and a window of route images
        limit = self.mem_pointer + self.window
        wrsims = rmf(query_img, self.route_images[self.mem_pointer:limit], self.matcher, self.deg_range, self.deg_step)

        # Holds the best rot. match between the query image and route images
        wind_sims = []
        # Recovered headings for the current image
        wind_headings = []
        for rsim in wrsims:
            # get best similarity match adn index w.r.t degrees
            index = self.argminmax(rsim)
            wind_sims.append(rsim[index])
            wind_headings.append(self.degrees[index])

        # Save the best degree match for window similarities
        self.window_sims.append(wind_sims)
        # append the rsims of all window route images for that query image
        self.logs.append(wrsims)
        # find best image match and heading
        index = self.argminmax(wind_sims)
        heading = wind_headings[index]
        self.recovered_heading.append(heading)
        # Update memory pointer
        self.mem_pointer += index
        self.matched_index_log.append(self.mem_pointer)

        # recovered_heading.append(sum(wind_headings)/len(wind_headings))

        if dynamic_window:
            best = wind_sims[index]
            self.update_window(best)

        return heading

    def update_window(self, best):
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window < 15:
            self.window += 2
        # elif self.window > 20:
        #     self.window -= 2
        else:
            self.window -= 2
        self.prev_match = best
        upper = int(self.window/2)
        lower = self.window - upper
        return upper, lower

    def navigate(self, query_imgs):
        assert isinstance(query_imgs, list)

        upper = int(self.window/2)
        lower = self.window - upper
        mem_pointer = upper
        flimit = self.window
        blimit = 0
        # For every query image
        for query_img in query_imgs:

            # get the rotational similarities between a query image and a window of route images
            wrsims = rmf(query_img, self.route_images[blimit:flimit], self.matcher, self.deg_range, self.deg_step)

            # Holds the best rot. match between the query image and route images
            wind_sims = []
            # Recovered headings for the current image
            wind_headings = []
            for rsim in wrsims:
                # get best similarity match adn index w.r.t degrees
                index = self.argminmax(rsim)
                wind_sims.append(rsim[index])
                wind_headings.append(self.degrees[index])

            # Save the best degree match for window similarities
            self.window_sims.append(wind_sims)
            # append the rsims of all window route images for that current image
            self.logs.append(wrsims)
            index = self.argminmax(wind_sims)
            # self.recovered_heading.append(wind_headings[index])
            # The heading ins the window average
            self.recovered_heading.append(np.mean(wind_sims))

            # Update memory pointer
            change = index - upper
            mem_pointer += change
            self.matched_index_log.append(mem_pointer)

            # Update the bounds of the window
            flimit = mem_pointer + upper
            blimit = mem_pointer - lower
            if flimit > self.route_end:
                flimit = self.route_end
                mem_pointer = self.route_end - upper
            if blimit <= 0:
                blimit = mem_pointer
                flimit = mem_pointer + self.window
                mem_pointer = mem_pointer + upper
            self.window_log.append([blimit, flimit])

            # Change the pointer and bounds for an adaptive window.
            if self.adaptive:
                upper, lower = self.update_window(wind_sims[index])


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

    def __init__(self, route_images, matching, deg_range=(0, 360), degree_shift=1):
        self.route_end = len(route_images)
        self.route_images = route_images
        self.deg_range = deg_range
        self.degree_shift = degree_shift
        self.degrees = np.arange(*deg_range)
        self.recovered_heading = []
        self.logs = []
        self.window_log = []
        self.matched_index_log = []
        self.confidence = [1] * self.route_end
        self.window_sims = []
        self.CMA = []
        matchers = {'corr': cor_dist, 'rmse': rmse, 'mae':mae}
        self.matcher = matchers.get(matching)
        if not self.matcher:
            raise Exception('Non valid matcher method name')
        self.argminmax = np.argmin
        self.prev_match = 0.0

    def navigate(self, w_g_imgs, window=10, mem_pointer=0):
        no_of_test_imgs = len(w_g_imgs)
        end_route = False
        current_img_pointer = -1
        # For every query sequence
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

            # get the rotational similarities between a query imag seq and a window of route images
            wrsims = pair_rmf(current_imgs, self.route_images[mem_pointer:limit], self.matcher, self.deg_range, self.deg_step)

            for rsim in wrsims:
                # get best similarity match for degrees and its index(heading)
                index = self.argminmax(rsim)
                wind_sims.append(rsim[index])
                wind_headings.append(self.degrees[index])

            # Save the best degree match for window similarities
            self.window_sims.append(wind_sims)
            # append the rsims of all window route images for that current image sequence
            self.logs.append(wrsims)
            index = self.argminmax(wind_sims)
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