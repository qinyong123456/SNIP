import os
import itertools
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class Dataset(object):
    def __init__(self, datasource, path_data, **kwargs):
        self.datasource = datasource
        self.path_data = path_data
        self.rand = np.random.RandomState(9)
        
        if self.datasource == 'mnist':
            self.num_classes = 10
            # 加载MNIST数据集
            train_dataset = datasets.MNIST(
                root=path_data,
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            test_dataset = datasets.MNIST(
                root=path_data,
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
            
            # 转换为NumPy数组
            train_images = train_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
            train_labels = train_dataset.targets.numpy()
            test_images = test_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
            test_labels = test_dataset.targets.numpy()
            
            self.dataset = {
                'train': {'input': train_images, 'label': train_labels},
                'test': {'input': test_images, 'label': test_labels}
            }
            
        elif self.datasource == 'cifar-10':
            self.num_classes = 10
            # 加载CIFAR-10数据集
            train_dataset = datasets.CIFAR10(
                root=path_data,
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            test_dataset = datasets.CIFAR10(
                root=path_data,
                train=False,
                download=True,
                transform=transforms.ToTensor()
            )
            
            # 转换为NumPy数组
            train_images = np.transpose(np.array(train_dataset.data), (0, 3, 1, 2)) / 255.0
            train_labels = np.array(train_dataset.targets)
            test_images = np.transpose(np.array(test_dataset.data), (0, 3, 1, 2)) / 255.0
            test_labels = np.array(test_dataset.targets)
            
            self.dataset = {
                'train': {'input': train_images, 'label': train_labels},
                'test': {'input': test_images, 'label': test_labels}
            }
            
        else:
            raise NotImplementedError
            
        # 分割训练集为训练集和验证集
        self.split_dataset('train', 'val', int(self.dataset['train']['input'].shape[0] * 0.1), self.rand)
        self.num_example = {k: self.dataset[k]['input'].shape[0] for k in self.dataset.keys()}
        self.example_generator = {
            'train': self.iterate_example('train'),
            'val': self.iterate_example('val'),
            'test': self.iterate_example('test', shuffle=False),
        }

    def iterate_example(self, mode, shuffle=True):
        epochs = itertools.count()
        for i in epochs:
            example_ids = list(range(self.num_example[mode]))
            if shuffle:
                self.rand.shuffle(example_ids)
            for example_id in example_ids:
                yield {
                    'input': self.dataset[mode]['input'][example_id],
                    'label': self.dataset[mode]['label'][example_id],
                    'id': example_id,
                }

    def get_next_batch(self, mode, batch_size):
        inputs, labels, ids = [], [], []
        for i in range(batch_size):
            example = next(self.example_generator[mode])
            inputs.append(example['input'])
            labels.append(example['label'])
            ids.append(example['id'])
        return {
            'input': np.asarray(inputs),
            'label': np.asarray(labels),
            'id': np.asarray(ids),
        }

    def generate_example_epoch(self, mode):
        example_ids = range(self.num_example[mode])
        for example_id in example_ids:
            yield {
                'input': self.dataset[mode]['input'][example_id],
                'label': self.dataset[mode]['label'][example_id],
                'id': example_id,
            }

    def split_dataset(self, source, target, number, rand):
        keys = ['input', 'label']
        indices = list(range(self.dataset[source]['input'].shape[0]))
        rand.shuffle(indices)
        ind_target = indices[:number]
        ind_remain = indices[number:]
        self.dataset[target] = {k: self.dataset[source][k][ind_target] for k in keys}
        self.dataset[source] = {k: self.dataset[source][k][ind_remain] for k in keys}
