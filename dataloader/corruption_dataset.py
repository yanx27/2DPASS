import os
import yaml
import numpy as np

from torch.utils import data

def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


class SemanticKITTIC(data.Dataset):
    def __init__(self, config, data_path, corruption, num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.corruption = corruption
        self.imageset = 'val'
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.im_idx = []
        self.im_idx += absoluteFilePaths('/'.join([data_path.replace('sequences', 'SemanticKITTI-C'), corruption, 'velodyne']), num_vote)
        self.im_idx = sorted(self.im_idx)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))

        origin_len = len(raw_data)
        raw_data = raw_data[:, :4]
        points = raw_data[:, :3]

        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.uint32).reshape((-1, 1))
        instance_label = annotated_data >> 16
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        if self.config['dataset_params']['ignore_label'] != 0:
            annotated_data -= 1
            annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len

        return data_dict, self.im_idx[index]