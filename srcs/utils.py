import torch
import torch.nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import math
import json
import os
import logger
from shapely.geometry import Point, Polygon
import copy as cp
def transform_label2metric(label, geometry):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in label map space
    :return: numpy array of shape [..., 2] of the same coordinates in metric space
    '''

    x_grid_size, y_grid_size, _ = get_discretization_from_geom(geometry, False)
    base_height = geometry["label_shape"][0]
    metric = np.copy(label)
    metric[..., 1] = metric[..., 1] - (base_height/2.0)
    metric[..., 1] = metric[..., 1] * y_grid_size 
    
    
    metric[..., 0] = metric[..., 0] * x_grid_size + geometry["W1"] 
    return metric

def transform_metric2label(metric, geometry):
    '''
    :param label: numpy array of shape [..., 2] of coordinates in metric space
    :return: numpy array of shape [..., 2] of the same coordinates in label_map space
    '''
    label = np.copy(metric)
    x_grid_size, y_grid_size, _ = get_discretization_from_geom(geometry, False)
    base_height = geometry["label_shape"][0]



    label[..., 0] = (label[..., 0] - geometry["W1"]) / x_grid_size
    label[..., 1] =  (label[..., 1] )/ y_grid_size
    label[..., 1] = label[..., 1] + (base_height/2.0)

    return label

def maskFOV_on_BEV(shape, fov=88.0):

    height = shape[0]
    width = shape[1]


    fov = fov / 2

    x = np.arange(width)
    y = np.arange(-height//2, height//2)

    xx, yy = np.meshgrid(x, y)
    angle = np.arctan2(yy, xx) * 180 / np.pi

    in_fov = np.abs(angle) < fov
    in_fov = torch.from_numpy(in_fov.astype(np.float32))

    return in_fov
from datetime import datetime
def get_logger(config, mode='train'):
    folder = os.path.join('logs', config['name'], mode, datetime.now().strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(folder):
        os.makedirs(folder)
    return logger.Logger(folder)

def get_discretization_from_geom(geom, input_layer=True):
    ''' We assume the input shape has '''
    if input_layer:
        xy_grid_shape = geom["input_shape"]
    else:
        xy_grid_shape = geom["label_shape"]        

    dy = (geom["L2"] - geom["L1"]) / (1.0 * xy_grid_shape[0])
    dx = (geom["W2"] - geom["W1"]) / (1.0 * xy_grid_shape[1])


    
    dz = (geom["H2"] - geom["H1"]) / (1.0 * (geom["input_shape"][2] - 1))

    return dx, dy, dz

def get_bev(velo_array, label_list = None, scores = None, geometry=None):
    ''' Compute the intensity BEV matrix and plot
        velo_array: [W,L,H+1] discretized grid of velodyne where (0,0) starts a L1, W1 
        label_list: list of BEV corners (in metric Velo frame)

        returns:
            intensity: [H,W,3] RGB intensity map in input dimensions
    
    '''
    
    map_height = velo_array.shape[0]
    intensity = np.zeros((velo_array.shape[0], velo_array.shape[1], 3), dtype=np.uint8)   
     # val = 1 - velo_array[::-1, :, -1]

    if False: 
        max_intensity = np.max(velo_array[::-1, :, :-1], axis=2 )
    
        val = (1 - max_intensity) * 255
        intensity[:, :, 0] = val
        intensity[:, :, 1] = val
        intensity[:, :, 2] = val
    
 
    # FLip in the x direction

        dx, dy, dz = get_discretization_from_geom(geometry, input_layer = True)  

        if label_list is not None:
            for corners in label_list:
                plot_corners = np.zeros(shape=corners.shape)
                # Convert to Pixels
                plot_corners[:, 0] = (corners[:, 0] - geometry["W1"])/ dx
                plot_corners[:, 1] = map_height - (corners[:, 1] / dy + int(map_height//2))

                plot_corners_int = plot_corners.astype(int).reshape((-1, 1, 2))
                cv2.polylines(intensity, [plot_corners_int], True, (255, 0, 0), 2)
                cv2.line(intensity, tuple(plot_corners_int[2, 0]), tuple(plot_corners_int[3, 0]), (0, 0, 255), 3)

    return intensity

def plot_bev(velo_array, label_list = None, scores = None, window_name='GT', save_path=None, geom=None):
    '''
    Plot a Birds Eye View Lidar and Bounding boxes (Using OpenCV!)
    The heading of the vehicle is marked as a red line
        (which connects front right and front left corner)

    :param velo_array: a 2d velodyne points
    :param label_list: a list of numpy arrays of shape [4, 2], which corresponds to the 4 corners' (x, y)
    The corners should be in the following sequence:
    rear left, rear right, front right and front left
    :param window_name: name of the open_cv2 window
    :return: None
    '''

    intensity = get_bev(velo_array, label_list, scores, geom)
    
    if save_path != None:
        print(save_path)
        cv2.imwrite(save_path, intensity)
        cv2.waitKey(0)
    else:
        cv2.imshow(window_name, intensity)
        cv2.waitKey(3)

    return intensity

def plot_label_map(label_map):
    plt.figure()
    plt.imshow(label_map[::-1, :])
    plt.show()

def plot_pr_curve(precisions, recalls, legend, name='PRCurve'):

    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, ".")
    ax.set_title("Precision Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend([legend], loc='upper right')
    path = os.path.join("Figures", name)
    fig.savefig(path)
    print("PR Curve saved at", path)


def get_points_in_a_rotated_box_metric(corners, dx, dy):
    ''' Return a list of xy points within the box'''
    points = []
    minx = np.min(corners[:, 0])
    maxx = np.max(corners[:, 0])
    miny = np.min(corners[:,1])
    maxy = np.max(corners[:,1])
    corners_pts = [(corners[i, 0], corners[i, 1]) for i in range(4)]
    bev_shape = Polygon(corners_pts)

    for x in np.arange(minx, maxx, dx):
        for y in np.arange(miny, maxy, dy):
            pt_shape = Point(x,y)
            if bev_shape.contains(pt_shape) :
                points.append((x,y))

    
    return points

def get_points_in_a_rotated_box(corners, label_shape=[200, 175], xmin=0, ymin=0, dx = 1.0, dy=1.):
    def minY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y0 is lowest
            return int(math.floor(y0))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # lowest point is at left edge of pixel column
            return int(math.floor(y0 + m * (x - x0)))
        else:
            # lowest point is at right edge of pixel column
            return int(math.floor(y0 + m * ((x + 1.0) - x0)))


    def maxY(x0, y0, x1, y1, x):
        if x0 == x1:
            # vertical line, y1 is highest
            return int(math.ceil(y1))

        m = (y1 - y0) / (x1 - x0)

        if m >= 0.0:
            # highest point is at right edge of pixel column
            return int(math.ceil(y0 + m * ((x + 1.0) - x0)))
        else:
            # highest point is at left edge of pixel column
            return int(math.ceil(y0 + m * (x - x0)))


    # view_bl, view_tl, view_tr, view_br are the corners of the rectangle
    view = [(corners[i, 0], corners[i, 1]) for i in range(4)]

    pixels = []

    # find l,r,t,b,m1,m2
    l, m1, m2, r = sorted(view, key=lambda p: (p[0], p[1]))
    b, t = sorted([m1, m2], key=lambda p: (p[1], p[0]))

    lx, ly = l
    rx, ry = r
    bx, by = b
    tx, ty = t
    m1x, m1y = m1
    m2x, m2y = m2

    xmin = 0
    ymin = 0
    xmax = label_shape[1]
    ymax = label_shape[0]

    # inward-rounded integer bounds
    # note that we're clamping the area of interest to (xmin,ymin)-(xmax,ymax)
    lxi = max(int(math.ceil(lx)), xmin)
    rxi = min(int(math.floor(rx)), xmax)
    byi = max(int(math.ceil(by)), ymin)
    tyi = min(int(math.floor(ty)), ymax)

    x1 = lxi
    x2 = rxi

    for x in range(x1, x2):
        xf = float(x)

        if xf < m1x:
            # Phase I: left to top and bottom
            y1 = minY(lx, ly, bx, by, xf)
            y2 = maxY(lx, ly, tx, ty, xf)

        elif xf < m2x:
            if m1y < m2y:
                # Phase IIa: left/bottom --> top/right
                y1 = minY(bx, by, rx, ry, xf)
                y2 = maxY(lx, ly, tx, ty, xf)

            else:
                # Phase IIb: left/top --> bottom/right
                y1 = minY(lx, ly, bx, by, xf)
                y2 = maxY(tx, ty, rx, ry, xf)

        else:
            # Phase III: bottom/top --> right
            y1 = minY(bx, by, rx, ry, xf)
            y2 = maxY(tx, ty, rx, ry, xf)

        y1 = max(y1, byi)
        y2 = min(y2, tyi)
        for y in range(y1, y2):
            pixels.append((x, y))

    return pixels

def load_config(exp_name, parent_dir=""):
    """ Loads the configuration file

     Args:
         path: A string indicating the path to the configuration file
     Returns:
         config: A Python dictionary of hyperparameter name-value pairs
         learning rate: The learning rate of the optimzer
         batch_size: Batch size used during training
         num_epochs: Number of epochs to train the network for
         target_classes: A list of strings denoting the classes to
                        build the classifer for
     """
    path = os.path.join(parent_dir, 'experiments', exp_name, 'config.json')
    with open(path) as file:
        config = json.load(file)

    assert config['name']==exp_name

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]

    return config, learning_rate, batch_size, max_epochs

def get_model_name(config, epoch=None, parent_dir=""):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        name: Name of ckpt
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    # path = "model_"
    # path += "epoch{}_".format(config["max_epochs"])
    # path += "bs{}_".format(config["batch_size"])
    # path += "lr{}".format(config["learning_rate"])

    name = config['name']
    if epoch is None:
        epoch = config['resume_from']

    folder = os.path.join(parent_dir, "experiments", name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = os.path.join(folder, str(epoch)+"epoch")
    return path

def writefile(config, filename, value):
    path = os.path.join('experiments', config['name'], filename)
    with open(path, 'a') as f:
        f.write(value)

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())

if __name__ == "__main__":
    maskFOV_on_BEV(0)
