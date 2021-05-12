from source.utils import mae, rmse, cor_dist, rmf, pair_rmf, cos_sim, mean_angle
import numpy as np


class SequentialPerfectMemory:

    def __init__(self, route_images, matching, deg_range=(0, 360), degree_shift=1, window=20):
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
        self.best_sims = []
        self.window_headings = []
        self.CMA = []
        # Matching variables
        matchers = {'corr': cor_dist, 'rmse': rmse, 'mae': mae}
        self.matcher = matchers.get(matching)
        if not self.matcher:
            raise Exception('Non valid matcher method name')
        self.argminmax = np.argmin
        self.prev_match = 0.0

        # Window parameters
        if window < 0:
            self.adaptive = True
        else:
            self.adaptive = False
        self.mem_pointer = 0
        self.window = abs(window)
        self.blimit = 0
        self.flimit = self.window

        # Adaptive window variables
        self.min_window = 10
        self.window_margin = 5
        self.deg_diff = 5
        self.upper = int(round(window/2))
        self.lower = window - self.upper
        self.agreement_thresh = 0.9

    def reset_window(self, pointer):
        self.update_pointer(pointer)

    def get_heading(self, query_img):
        '''
        Recover the heading given a query image
        :param query_img:
        :return:
        '''
        # get the rotational similarities between a query image and a window of route images
        wrsims = rmf(query_img, self.route_images[self.blimit:self.flimit], self.matcher, self.deg_range, self.deg_step)
        self.window_log.append([self.blimit, self.flimit])
        # Holds the best rot. match between the query image and route images
        wind_sims = []
        # Recovered headings for the current image
        wind_headings = []
        # get best similarity match adn index w.r.t degrees
        indices = self.argminmax(wrsims, axis=1)
        for i, idx in enumerate(indices):
            wind_sims.append(wrsims[i, idx])
            wind_headings.append(self.degrees[idx])

        # Save the best degree and sim for window similarities
        self.window_sims.append(wind_sims)
        self.window_headings.append(wind_headings)
        # append the rsims of all window route images for that query image
        self.logs.append(wrsims)
        # find best image match and heading
        index = int(self.argminmax(wind_sims))
        self.best_sims.append(wind_sims[index])
        heading = wind_headings[index]
        self.recovered_heading.append(heading)
        # Update memory pointer
        self.update_pointer(index)

        self.matched_index_log.append(self.mem_pointer)

        if self.adaptive:
            best = wind_sims[index]
            self.dynamic_window_sim(best)
            self.check_w_size()
        return heading

    def update_pointer(self, index):
        '''
        Update the mem pointer to the back of the window
        mem_pointer = blimit
        :param index:
        :return:
        '''
        self.mem_pointer += index
        if self.mem_pointer + self.window > self.route_end:
            self.mem_pointer = self.blimit + index
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window
        else:
            self.blimit = self.mem_pointer
            self.flimit = self.mem_pointer + self.window

    def update_mid_pointer(self, index):
        '''
        Update the mem pointer to the middle of the window
        :param index:
        :return:
        '''
        # Update memory pointer
        change = index - self.upper
        self.mem_pointer += change
        self.matched_index_log.append(self.mem_pointer)

        # Update the bounds of the window
        self.flimit = self.mem_pointer + self.upper
        self.blimit = self.mem_pointer - self.lower
        if self.flimit > self.route_end:
            self.flimit = self.route_end
            self.blimit = self.route_end - self.window
        if self.blimit <= 0:
            self.blimit = self.mem_pointer
            self.flimit = self.mem_pointer + self.window

    def check_w_size(self):
        self.window = self.route_end if self.window > self.route_end else self.window

    def get_agreement(self, window_headings):
        a = np.full(len(window_headings), 1)
        return cos_sim(a, window_headings)

    def consensus_heading(self, wind_headings, h):
        '''
        Calculates the agreement of the window headings.
        If the agreement is above the threshold the best heading is used
        otherwise the last heading is used.
        :param wind_headings:
        :param h:
        :return:
        '''
        if self.get_agreement(wind_headings) >= self.agreement_thresh:
            self.recovered_heading.append(h)
        elif len(self.recovered_heading) > 0:
            self.recovered_heading.append(self.recovered_heading[-1])
        else:
            self.recovered_heading.append(h)

    def average_heading2(self, h):
        '''
        Calculates the average of the last window heading and the current window heading
        :param h:
        :return:
        '''
        if len(self.recovered_heading) > 0:
            self.recovered_heading.append(mean_angle([h, self.recovered_heading[-1]]))
        else:
            self.recovered_heading.append(h)

    def average_headings(self, wind_heading):
        '''
        Calculates the average of all window headings
        :param wind_heading:
        :return:
        '''
        self.recovered_heading.append(mean_angle(wind_heading))

    def dynamic_window_sim(self, best):
        '''
        Change the window size depending on the best img match gradient.
        If the last best img sim > the current best img sim the window grows
        and vice versa
        :param best:
        :return:
        '''
        # Dynamic window adaptation based on match gradient.
        if best > self.prev_match or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin
        self.prev_match = best
        # upper = int(self.window/2)
        # lower = self.window - upper
        # return upper, lower

    def dynamic_window_h2(self, h):
        '''
        Change the window size depending on the best heading gradient.
        If the difference between the last heading and the current heading is > self.deg.diff
        then the window grows and vice versa
        :param h:
        :return:
        '''
        diff = abs(h - self.recovered_heading[-1])
        if diff > self.deg_diff or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin

    def dynamic_window_h(self, wind_headings):
        '''
        The window grows if the window headings disagree and vice versa
        :param wind_headings:
        :return:
        '''
        if self.get_agreement(wind_headings) <= self.agreement_thresh or self.window <= self.min_window:
            self.window += self.window_margin
        else:
            self.window -= self.window_margin

    def navigate(self, query_imgs):
        assert isinstance(query_imgs, list)

        # upper = int(self.window/2)
        # lower = self.window - upper
        # mem_pointer = upper
        mem_pointer = 0
        flimit = self.window
        blimit = 0
        self.window_log.append([blimit, flimit])
        # For every query image
        for query_img in query_imgs:

            # get the rotational similarities between a query image and a window of route images
            wrsims = rmf(query_img, self.route_images[blimit:flimit], self.matcher, self.deg_range, self.deg_step)
            self.window_log.append([blimit, flimit])
            # Holds the best rot. match between the query image and route images
            wind_sims = []
            # Recovered headings for the current image
            wind_headings = []
            # get best similarity match adn index w.r.t degrees
            indices = self.argminmax(wrsims, axis=1)
            for i, idx in enumerate(indices):
                wind_sims.append(wrsims[i, idx])
                wind_headings.append(self.degrees[idx])

            # Save the best degree and sim for each window similarities
            self.window_sims.append(wind_sims)
            self.window_headings.append(wind_headings)
            # append the rsims of all window route images for that current image
            self.logs.append(wrsims)
            index = self.argminmax(wind_sims)
            self.best_sims.append(wind_sims[index])
            h = wind_headings[index]
            self.recovered_heading.append(h)
            # self.average_heading2(h)
            # self.average_headings(wind_headings)
            # self.consensus_heading(wind_headings, h)

            mem_pointer += index
            if mem_pointer + self.window > self.route_end:
                mem_pointer = blimit + index
                flimit = self.route_end
                blimit = self.route_end - self.window
            else:
                blimit = mem_pointer
                flimit = mem_pointer + self.window

            self.matched_index_log.append(mem_pointer)
            # self.window_log.append([blimit, flimit])


            # Change the pointer and bounds for an adaptive window.
            if self.adaptive:
                self.dynamic_window_sim(wind_sims[index])
                # self.dynamic_window_h2(h)
                # self.dynamic_window_h(wind_headings)

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

        return self.recovered_heading, self.window_log

    def get_index_log(self):
        return self.matched_index_log

    def get_window_log(self):
        return self.window_log

    def get_confidence(self):
        return self.confidence

    def get_window_sims(self):
        return self.window_sims

    def get_best_sims(self):
        return self.best_sims

    def get_window_headings(self):
        return self.window_headings

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