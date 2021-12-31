import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return transforms.Compose([
        transforms.RandomCrop(crop_size, pad_if_needed=True),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])


display_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor()
])


class TrainDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDataset, self).__init__()
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]).convert('RGB'))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(ValDataset, self).__init__()
        self.crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor
        self.image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index]).convert('RGB')
        
        lr_scale = transforms.Resize(self.crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = transforms.Resize(self.crop_size, interpolation=Image.BICUBIC)
        hr_image = transforms.CenterCrop(self.crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return to_tensor(lr_image), to_tensor(hr_restore_img), to_tensor(hr_image)

    def __len__(self):
        return len(self.image_filenames)