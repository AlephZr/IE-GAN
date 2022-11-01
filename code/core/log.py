import os
import time


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option

        # create a logging file to store time
        self.time_log_name = os.path.join(opt.savefolder, 'time_log.txt')
        with open(self.time_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Wall-clock Time (%s) ================\n' % now)

        # create a logging file to store time
        self.score_log_name = os.path.join(opt.savefolder, 'score_log.txt')
        with open(self.score_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Scores (%s) ================\n' % now)

    def record_time(self, gen, start_time):
        """Save wall-clock time to a file

        Parameters:
            start_time(float) -- Start time
        """
        end_time = time.time()
        with open(self.time_log_name, 'a') as log_file:
            log_file.write('Gen: {0}, Running time: {1} Seconds\n'.format(gen, end_time - start_time))

    def print_current_scores(self, epoch, iters, scores):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            scores (OrderedDict) -- training losses stored in the format of (name, float) pairs
        """
        message = '(epoch: %d, giters: %d) ' % (epoch, iters)
        for k, v in scores.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # print the message
        with open(self.score_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message