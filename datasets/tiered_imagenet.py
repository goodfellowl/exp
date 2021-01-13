import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



# def load_data(file):
#     try:
#         with open(file, 'rb') as fo:
#             data = pickle.load(fo)
#         return data
#     except:
#         with open(file, 'rb') as f:
#             u = pickle._Unpickler(f)
#             u.encoding = 'latin1'
#             data = u.load()
#         return data
        
class TieredImageNet(Dataset):

    def __init__(self, root_path='./datasets/data/tired_imagenet/', split='train', train_aug=None, rotate=False):
        # data = np.load(os.path.join(root_path, '{}_images.npz'.format(split)), allow_pickle=True)['images']
        # data = data[:, :, :, ::-1]
       
        # labels = load_data(os.path.join(root_path, '{}_labels.pkl'.format(split)))['label']
        
        # with open(os.path.join(root_path, '{}_labels.pkl'.format(split)), 'rb') as f:
        #     label = pickle.load(f)['labels']

        # data = [Image.fromarray(x) for x in data]
        # min_label = min(label)
        # label = [x - min_label for x in label]
        # label = []
        # lb = -1
        # self.wnids = []
        # for wnid in labels:
        #     if wnid not in self.wnids:
        #         self.wnids.append(wnid)
        #         lb += 1
        #     label.append(lb)


        images_dir = os.path.join(root_path, '{}_images.npz'.format(split))
        labels_dir = os.path.join(root_path, '{}_labels.pkl'.format(split))
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            try:
                with open(labels_dir) as f:
                    label_data = pkl.load(f)
                    label_specific = label_data['label_specific']         
            except:
                with open(labels_dir, 'rb') as f:
                    label_data = pkl.load(f, encoding='bytes')
                    label_specific = label_data['label_specific']            
            print('read label data:{}'.format(len(label_specific)))
        else:
            print('no file exists!')
        
        labels = label_specific
        n_classes = np.max(labels)+1

        with np.load(images_dir, mmap_mode="r", encoding='latin1') as data:
            images = data["images"]
            print('read image data:{}'.format(images.shape))

        self.data = images
        self.label = labels
        self.n_classes = n_classes
        
        print("total {} class".format(self.n_classes))
        if rotate:
            self.rotation_labels = [0, 1, 2, 3]
        else:
            self.rotation_labels = []
        
        image_size = 84
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,  
        ])
    
        if train_aug == 'resize':
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif train_aug == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif train_aug is None:
            self.transform = self.default_transform
        # norm_params = {'mean': [0.485, 0.456, 0.406],
        #                'std': [0.229, 0.224, 0.225]}
        # if train_aug:
        #     image_size = 84
        #     self.transform = transforms.Compose([
        #         transforms.Resize(92),
        #         transforms.RandomResizedCrop(88),
        #         transforms.CenterCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        # else:
        #     image_size = 84
        #     self.transform = transforms.Compose([
        #         transforms.Resize(92),
        #         transforms.CenterCrop(image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
        #                              np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])


        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw
    
    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        if len(self.rotation_labels) > 1:
            image, label = self.data[index], self.label[index]
            image_90 = self.transform(Image.fromarray(self.rotate_img(image, 90)))
            image_180 = self.transform(Image.fromarray(self.rotate_img(image, 180)))
            image_270 = self.transform(Image.fromarray(self.rotate_img(image, 270)))
            image = self.transform(Image.fromarray(image))
            image = torch.stack([image, image_90, image_180, image_270])
            return image, torch.ones(len(self.rotation_labels), dtype=torch.long)*int(label), torch.LongTensor(self.rotation_labels)
        else:
            return self.transform(Image.fromarray(self.data[index])), torch.tensor(self.label[index]).long()



# from samplers import CategoriesSampler
# from torch.utils.data import DataLoader
# if __name__ == '__main__':
    
#     train_dataset = TieredImageNet(split='test')
#     train_sampler = CategoriesSampler(train_dataset.label, 1, 5, 6)
#     train_loader = DataLoader(train_dataset, batch_sampler=train_sampler,
#                               num_workers=0, pin_memory=True)
#     for data,label in train_loader:
#         print(data.shape)
#         print(label)
#         print(label.shape)