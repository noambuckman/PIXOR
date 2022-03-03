from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""

        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        images = np.array(images)
        if len(images.shape) == 4:
            images = images.transpose(0, 3, 1, 2) #move color channel
        else:
            images = images.transpose(2, 0, 1) #move color channel
        images = images/256.0
        self.writer.add_images(tag, images, step)
        self.writer.flush()

    def config_summary(self, config):
        config_string = ""
        for k,v in config:
            config_string += "%s : %s \n"%(k, v)
        
        self.writer.add_text("configs", config_string)
 
    # def histo_summary(self, tag, values, step, bins=1000):
    #     """Log a histogram of the tensor of values."""

    #     # Create a histogram using numpy
    #     counts, bin_edges = np.histogram(values, bins=bins)

    #     # Fill the fields of the histogram proto
    #     hist = tf.HistogramProto()
    #     hist.min = float(np.min(values))
    #     hist.max = float(np.max(values))
    #     hist.num = int(np.prod(values.shape))
    #     hist.sum = float(np.sum(values))
    #     hist.sum_squares = float(np.sum(values ** 2))

    #     # Drop the start of the first bin
    #     bin_edges = bin_edges[1:]

    #     # Add bin edges and counts
    #     for edge in bin_edges:
    #         hist.bucket_limit.append(edge)
    #     for c in counts:
    #         hist.bucket.append(c)

    #     # Create and write Summary
    #     summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    #     self.writer.add_summary(summary, step)
    #     self.writer.flush()