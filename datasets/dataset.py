

def make(name, s, aug=None, rot=False):

    if name == 'mini':
        from datasets.mini_imagenet import MiniImagenet
        dataset = MiniImagenet(split=s, train_aug=aug, rotate=rot)
    elif name == 'tiered':
        from datasets.tiered_imagenet import TieredImageNet
        dataset = TieredImageNet(split=s, train_aug=aug, rotate=rot)
    
    return dataset
