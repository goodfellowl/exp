
""" Dataloader for mini_imagenet datasets. """
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch




class MiniImagenet(Dataset):
    """The class to load the dataset"""
    def __init__(self, root_path='./datasets/data/mini_imagenet/data', split='train', train_aug=None, rotate=False):
        # Set the path according to train, val and test
        if split == 'train':
            THE_PATH = os.path.join(root_path, 'train')
            label_list = os.listdir(THE_PATH)
        elif split == 'test':
            THE_PATH = os.path.join(root_path, 'test')
            label_list = os.listdir(THE_PATH)
        elif split == 'val':
            THE_PATH = os.path.join(root_path, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong spilt.')

        # Generate empty list for data and label
        data = []
        label = []

        # Get folders' name
        folders = [os.path.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(os.path.join(THE_PATH, the_label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(os.path.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.n_classes = len(set(label))
        if rotate:
            self.rotation_labels = [0, 1, 2, 3]
        else:
            self.rotation_labels = []
        
        print('read label data:{}'.format(len(self.label)))
        print('read image data:{}'.format(len(self.data)))
        print("total {} class".format(self.n_classes))


        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        # normalize = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                    #  np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
        image_size = 84
        self.default_transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,  
        ])
        
        if train_aug == 'resize':
            self.transform = transforms.Compose([
                transforms.Resize(92),
                 # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
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
    
    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = Image.open(path).convert('RGB')
        if len(self.rotation_labels) > 1:
            image_90 = self.transform(Image.fromarray(self.rotate_img(image, 90)))
            image_180 = self.transform(Image.fromarray(self.rotate_img(image, 180)))
            image_270 = self.transform(Image.fromarray(self.rotate_img(image, 270)))
            image = self.transform(image)
            image = torch.stack([image, image_90, image_180, image_270])
            return image, torch.ones(len(self.rotation_labels), dtype=torch.long)*int(label), torch.LongTensor(self.rotation_labels)
        else:
            return self.transform(image), label
    

# from samplers import CategoriesSampler
# from torch.utils.data import DataLoader
# import os
# if __name__ == '__main__':
#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     train_dataset = MiniImagenet(split='train', train_aug='resize')
    
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
#     for data,label in train_loader:
       
#         print(data.shape)
       
#         print(label)
       