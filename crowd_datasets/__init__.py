import torchvision.transforms as standard_transforms
from .SHHA.SHHA import SHHA
from .SHHA.SHHA_New import SHHA_New

# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        return loading_data_sha(args)
    if args.dataset_file == 'SHHA_New':
        return loading_data_sha_new(args)

    return None


def loading_data_sha(args):
    data_root = args.data_root
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    train_set = SHHA(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = SHHA(data_root, train=False, transform=transform)

    return train_set, val_set


def loading_data_sha_new(args):
    data_root = args.data_root
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    train_set = SHHA_New(data_root, train=True, transform=transform, patch=True, flip=True)
    # create the validation dataset
    val_set = SHHA_New(data_root, train=False, transform=transform)

    return train_set, val_set


# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor