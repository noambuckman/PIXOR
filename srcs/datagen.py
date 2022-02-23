'''
Load pointcloud/labels from the KITTI dataset folder
'''
import os.path
import numpy as np
import time
import torch
import ctypes
from utils import load_config, get_discretization_from_geom, plot_bev, get_points_in_a_rotated_box, plot_label_map, transform_metric2label, transform_label2metric
from torch.utils.data import Dataset, DataLoader

#KITTI_PATH = '/home/autoronto/Kitti/object'
KITTI_PATH = '/mnt/ssd2/od/KITTI'
KITTI_PATH = '/home/data/kitti'
# KITTI_PATH = '/media/nbuckman/Samsung_S3/racecar_bags/kitti_rc_velodyne_20220202'
KITTI_PATH = 'data/kitti'
#KITTI_PATH = 'KITTI'

class KITTI(Dataset):
    
    # geometry = {
    #     'L1': -40.0,
    #     'L2': 40.0,
    #     'W1': 0.0,
    #     'W2': 70.0,
    #     'H1': -2.5,
    #     'H2': 1.0,
    #     'input_shape': (800, 700, 36),
    #     'label_shape': (200, 175, 7)
    # }

    # target_mean = np.array([0.008, 0.001, 0.202, 0.2, 0.43, 1.368])
    # target_std_dev = np.array([0.866, 0.5, 0.954, 0.668, 0.09, 0.111])


    def __init__(self, frame_range = 10000, use_npy=False, train=True, target_mean = None, target_std_dev=None, ignore_list=None):
        self.frame_range = frame_range
        self.velo = []
        self.use_npy = use_npy
        # self.LidarLib = ctypes.cdll.LoadLibrary('preprocess/LidarPreprocess.so')
        self.image_sets = self.load_imageset(train, ignore_list) # names

        self.target_mean = target_mean
        self.target_std_dev = target_std_dev

    def __len__(self):
        return len(self.image_sets)

    def __getitem__(self, item):
        scan = self.load_velo_scan(item)
        scan = torch.from_numpy(scan)
        label_map, _ = self.get_label(item)
        self.reg_target_transform(label_map)
        label_map = torch.from_numpy(label_map)
        scan = scan.permute(2, 0, 1)
        label_map = label_map.permute(2, 0, 1)
        return scan, label_map, item

    def reg_target_transform(self, label_map):
        '''
        Inputs are numpy arrays (not tensors!)
        :param label_map: [200 * 175 * 7] label tensor
        :return: normalized regression map for all non_zero classification locations
        '''
        cls_map = label_map[..., 0]
        reg_map = label_map[..., 1:]

        index = np.nonzero(cls_map)
        reg_map[index] = (reg_map[index] - self.target_mean)/self.target_std_dev


    def load_imageset(self, train, ignore_list=None):
        if ignore_list is None:
            ignore_list = []
        print("ignore list", ignore_list)
        path = KITTI_PATH
        if train:
            path = os.path.join(path, "train.txt")
        else:
            path = os.path.join(path, "val.txt")
        assert os.path.isfile(path), path
        with open(path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            names = []
            for line in lines[:-1]:
                if int(line[:-1]) < self.frame_range and line[:-1] not in ignore_list:
                    names.append(line[:-1])

            # Last line does not have a \n symbol
            last = lines[-1][:6]
            if int(last) < self.frame_range:
                names.append(last)
            # print(names[-1])
            print("There are {} images in txt file".format(len(names)))
            
            return names
    
    def interpret_kitti_label(self, bbox):
        w, h, l, y, z, x, yaw = bbox[8:15]
        y = -y
        yaw = - (yaw + np.pi / 2)
        
        return x, y, w, l, yaw
    
    def interpret_custom_label(self, bbox):
        w, l, x, y, yaw = bbox
        return x, y, w, l, yaw

    def get_corners(self, bbox):

        w, h, l, y, z, x, yaw = bbox[8:15]
        y = -y
        # manually take a negative s. t. it's a right-hand system, with
        # x facing in the front windshield of the car
        # z facing up
        # y facing to the left of driver

        yaw = -(yaw + np.pi / 2)
        #x, y, w, l, yaw = self.interpret_kitti_label(bbox)
        
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        reg_target = [np.cos(yaw), np.sin(yaw), x, y, w, l]

        return bev_corners, reg_target


    def update_label_map(self, map, bev_corners, reg_target):
        dx, dy, dz = get_discretization_from_geom(self.geometry, input_layer=False)
        
        label_corners = transform_metric2label(bev_corners, dx, dy, self.geometry['label_shape'][0])
        
        points = get_points_in_a_rotated_box(label_corners, self.geometry['label_shape'])
        for p in points:
            label_x = p[0]
            label_y = p[1]

            metric_x, metric_y = transform_label2metric(np.array(p), dx, dy, self.geometry['label_shape'][0])

            actual_reg_target = np.copy(reg_target)
            actual_reg_target[2] = reg_target[2] - metric_x
            actual_reg_target[3] = reg_target[3] - metric_y
            actual_reg_target[4] = np.log(reg_target[4])
            actual_reg_target[5] = np.log(reg_target[5])

            map[label_y, label_x, 0] = 1.0
            map[label_y, label_x, 1:7] = actual_reg_target


    def get_label(self, index):
        '''
        :param i: the ith velodyne scan in the train/val set
        :return: label map: <--- This is the learning target
                a tensor of shape 800 * 700 * 7 representing the expected output


                label_list: <--- Intended for evaluation metrics & visualization
                a list of length n; n =  number of cars + (truck+van+tram+dontcare) in the frame
                each entry is another list, where the first element of this list indicates if the object
                is a car or one of the 'dontcare' (truck,van,etc) object

        '''
        index = self.image_sets[index]
        f_name = (6-len(index)) * '0' + index + '.txt'
        label_path = os.path.join(KITTI_PATH, 'training', 'label_2', f_name)

        object_list = {'Car': 1, 'Truck':0, 'DontCare':0, 'Van':0, 'Tram':0}
        label_map = np.zeros(self.geometry['label_shape'], dtype=np.float32)
        label_list = []
        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    bbox.append(object_list[name])
                    bbox.extend([float(e) for e in entry[1:]])
                    if name == 'Car':
                        corners, reg_target = self.get_corners(bbox)
                        self.update_label_map(label_map, corners, reg_target)
                        label_list.append(corners)
        return label_map, label_list

    def get_rand_velo(self):
        import random
        rand_v = random.choice(self.velo)
        print("A Velodyne Scan has shape ", rand_v.shape)
        return random.choice(self.velo)

    def load_velo_scan(self, item):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = self.velo[item]
        assert os.path.isfile(filename), filename
        if self.use_npy:
            scan = np.load(filename[:-4]+'.npy')
        else:
            c_name = bytes(filename, 'utf-8')
            scan = np.zeros(self.geometry['input_shape'], dtype=np.float32)
            c_data = ctypes.c_void_p(scan.ctypes.data)
            self.LidarLib.createTopViewMaps(c_data, c_name)
            # scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            
        return scan

    def load_velo_scan2(self, item):
        filename = self.velo[item]
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)        
        
        return points      

    def load_velo(self):
        """Load velodyne [x,y,z,reflectance] scan data from binary files."""
        # Find all the Velodyne files

        velo_files = []
        for file in self.image_sets:
            file = '{}.bin'.format(file)
            velo_file_path = os.path.join(KITTI_PATH, 'training', 'velodyne', file)
            assert os.path.isfile(velo_file_path), velo_file_path

            velo_files.append(velo_file_path)

        print('Found ' + str(len(velo_files)) + ' Velodyne scans...')
        # Read the Velodyne scans. Each point is [x,y,z,reflectance]
        self.velo = velo_files

        print('done.')

    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def passthrough(self, velo):
        geom = self.geometry
        q = (geom['W1'] < velo[:, 0]) * (velo[:, 0] < geom['W2']) * \
            (geom['L1'] < velo[:, 1]) * (velo[:, 1] < geom['L2']) * \
            (geom['H1'] < velo[:, 2]) * (velo[:, 2] < geom['H2'])
        indices = np.where(q)[0]
        return velo[indices, :]

    def lidar_preprocess(self, scan):
        velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)
        intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
        velo = self.passthrough(scan)


        dx, dy, dz = get_discretization_from_geom(self.geometry, input_layer=True)

        for i in range(velo.shape[0]):
            x = int((velo[i, 1]-self.geometry['L1']) / dx)
            y = int((velo[i, 0]-self.geometry['W1']) / dy)
            z = int((velo[i, 2]-self.geometry['H1']) / dz)

            velo_processed[x, y, z] = 1
            velo_processed[x, y, -1] += velo[i, 3]
            intensity_map_count[x, y] += 1
        velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count,
                                             where=intensity_map_count != 0)
        return velo_processed


def get_data_loader(batch_size, use_npy, geometry=None, frame_range=10000, target_mean=None, target_std_dev=None, ignore_list=None):
    train_dataset = KITTI(frame_range, use_npy=use_npy, train=True, target_mean=target_mean, target_std_dev=target_std_dev, ignore_list=ignore_list)
    if geometry is not None:
        train_dataset.geometry = geometry
    train_dataset.load_velo()
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=3)
    val_dataset = KITTI(frame_range, use_npy=use_npy, train=False,  target_mean=target_mean, target_std_dev=target_std_dev, ignore_list=ignore_list)
    if geometry is not None:
        val_dataset.geometry = geometry
    val_dataset.load_velo()
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size * 4, num_workers=8)

    print("------------------------------------------------------------------")
    return train_data_loader, val_data_loader


def test0(id=25):
    config, _,_,_ = load_config("default")
    
    k = KITTI()
    k.geometry = config["geometry"]

    dx, dy, dz = get_discretization_from_geom(config["geometry"])
    
    k.load_velo()
    print(k.velo[id])
    tstart = time.time()
    scan = k.load_velo_scan(id)
    processed_v = k.lidar_preprocess(scan)
    label_map, label_list = k.get_label(id)
    print(label_list)
    print('time taken: %gs' %(time.time()-tstart))
    plot_bev(processed_v, label_list, geom=config["geometry"])
    plot_label_map(label_map[:, :, 6])

def test1(id=25):
    config, _,_,_ = load_config("default")
    
    k = KITTI(use_npy=True)
    k.geometry = config["geometry"]

    dx, dy, dz = get_discretization_from_geom(config["geometry"])

    k.load_velo()
    print(k.velo[id])
    tstart = time.time()


    processed_v = k.load_velo_scan(id)
    # processed_v = k.lidar_preprocess(scan)
    label_map, label_list = k.get_label(id)
    print(label_list)
    print('time taken: %gs' %(time.time()-tstart))
    plot_bev(processed_v, label_list, geom=config["geometry"])
    plot_label_map(label_map[:, :, 6])    


def view_multiple(n=100):
    for i in range(100):
        id = np.random.randint(0, 600)
        test1(id)
        input("Press Enter to continue...")        


def find_reg_target_var_and_mean_kitti(kitti):
    reg_targets = [[] for _ in range(6)]
    for i in range(len(kitti)):
        label_map, _ = kitti.get_label(i)
        car_locs = np.where(label_map[:, :, 0] == 1)
        for j in range(1, 7):
            map = label_map[:, :, j]
            reg_targets[j-1].extend(list(map[car_locs]))

    reg_targets = np.array(reg_targets)
    means = reg_targets.mean(axis=1)
    stds = reg_targets.std(axis=1)
    stds = stds + 0.01*np.ones(shape=stds.shape)
    np.set_printoptions(precision=3, suppress=True)
    print("Means", means)
    print("Stds", stds)
    return means, stds    

def find_reg_target_var_and_mean(geom=None):
    k = KITTI(use_npy=True, train=True)
    if geom is not None:
        k.geometry = geom
    reg_targets = [[] for _ in range(6)]
    for i in range(len(k)):
        label_map, _ = k.get_label(i)
        car_locs = np.where(label_map[:, :, 0] == 1)
        for j in range(1, 7):
            map = label_map[:, :, j]
            reg_targets[j-1].extend(list(map[car_locs]))
        # print(len(car_locs[0]))



    reg_targets = np.array(reg_targets)
    means = reg_targets.mean(axis=1)
    stds = reg_targets.std(axis=1)
    stds = stds + 0.01*np.ones(shape=stds.shape)
    np.set_printoptions(precision=3, suppress=True)
    print("Means", means)
    print("Stds", stds)
    return means, stds

def find_samples_without_labels(geom=None):
    sample_ids = []

    for train in [True, False]:
        k = KITTI(use_npy=True, train=train)
        if geom is not None:
            k.geometry = geom
            
        for i in range(len(k)):
            label_map, _ = k.get_label(i)
            car_locs = np.where(label_map[:, :, 0] == 1)
            if len(car_locs[0])==0:
                sample_ids.append(k.image_sets[i])
    print(sample_ids)
    return sample_ids


def preprocess_to_npy(train=True, geometry=None):
    k = KITTI(train=train)
    if geometry is not None:
        k.geometry = geometry
    
    k.load_velo()
    for item, name in enumerate(k.velo):
        scan = k.load_velo_scan2(item)
        scan = k.lidar_preprocess(scan)
        path = name[:-4] + '.npy'
        np.save(path, scan)
        print('Saved ', path)
    return

def test():
    # npy average time 0.31s
    # c++ average time 0.08s 4 workers
    batch_size = 3
    train_data_loader, val_data_loader = get_data_loader(batch_size, False)
    times = []
    tic = time.time()
    for i, (input, label_map, item) in enumerate(train_data_loader):
        toc = time.time()
        print(toc - tic)
        times.append(toc-tic)
        tic = time.time()
        #print("Entry", i)
        #print("Input shape:", input.shape)
        #print("Label Map shape", label_map.shape)
        if i == 20:
            break
    print("average preprocess time per image", np.mean(times)/batch_size)    

    print("Finish testing train dataloader")


if __name__=="__main__":
    # test0(id=25)
    config, _, _, _ = load_config("default")
    find_reg_target_var_and_mean(config["geometry"])

    
    # print("Processing Train")
    # preprocess_to_npy(True, geometry=config["geometry"])
    # print("Processing Val")
    # preprocess_to_npy(False, geometry=config["geometry"])
