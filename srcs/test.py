
from distutils.log import debug
import numpy as np
import time

from datagen import KITTI

# from model import PIXOR
from logger import get_logger
from utils import get_model_name, load_config, plot_bev, plot_label_map, plot_pr_curve, get_bev
# from postprocess import filter_pred, compute_matches, compute_ap
import matplotlib.pyplot as plt

from datagen import test0

if __name__ == "__main__":

    config, _,_,_ = load_config("default")
    id = 419
    id = 56
    k = KITTI(train=True, use_npy=False, geometry=config["geometry"])
    k.debug = True

    k.load_velo()
    # print(k.velo[id])
    # tstart = time.time()
    # scan = k.load_velo_scan(id)  # [x, y ,z ]

    label_map, label_list = k.get_label(id)
    print("BIN", k.image_sets[id])
    scan = k.scan_from_label(label_list)
    # scan, label_, item = k[id]

    print("LB", label_list)
    logger = get_logger(config, 'train')
    print(scan.shape)
    pred_image = get_bev(scan, label_list, None, geometry=config["geometry"])
    print("Max", np.min(pred_image))
    print("Min", np.max(pred_image))
    print(pred_image)
    print("Label List")
    print(label_list)
    print("pred image")
    print(pred_image.shape)
    # print(pred_image[25:30, 35:30, :])
    
    image_batch = np.zeros((1, pred_image.shape[0], pred_image.shape[1], pred_image.shape[2]))
    image_batch[0, :,:,:] = pred_image
    # image_batch[1, :, :, :] = pred_image
    logger.image_summary("imgs", image_batch, 0)



    print(scan.shape)
    plot_bev(scan, label_list, geom=config["geometry"])
    # plt.show()
    plot_label_map(label_map[:, :, 6])    
    # print(label_list)
    # print(label_map[94, 65, :])
    # print(label_map.shape)
    # filename = k.velo[id]
    # points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    # points = k.passthrough(points)
    
    # pos_px = label_map[:, :, 0] > 0
    # print("class pxs", pos_px.sum())
    # print("Prior %.02f"%(pos_px.sum()/(label_map.shape[0]*label_map.shape[1])))
    # plt.plot(points[:, 0], points[:,1], '.')
    # plt.show()
    # window = (scan[:,:,-1] ==1)
    # print(window.shape) 
    # for i in range(scan.shape[1]):
    #     for j in range(scan.shape[0]):
    #         if scan[j, i, -1]==1:    
    #             plt.plot(i, j, '.')
    # plt.show()
    # window = (scan[:,2]> - .5) * (scan[:, 1] > 0 )*(scan[:, 1] < 1.8 ) * (scan[:, 0] > -.1 )*(scan[:, 0] < .1 )
    # print(np.sum(window))
    # print(scan.shape)
    # print(scan[window, :2])
    # plt.plot(scan[window, 0], scan[window, 1], '.')
    # plt.show()
    # print(scan[window,:].shape)
    # processed_v = k.lidar_preprocess(scan)
    # print(processed_v.shape)
    # print(processed_v[398:402, 240:280, -1])

    # plot_bev(processed_v, label_list, geom=config["geometry"])
    # plot_label_map(label_map[:, :, 6])    

