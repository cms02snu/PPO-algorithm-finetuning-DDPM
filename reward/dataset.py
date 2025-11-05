import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DeepFakeDataset(Dataset):
    def __init__(self, real_img_dir, fake_img_dir, num_img=3000, num_noise=500, transform=None):
        self.real_img_dir = real_img_dir
        self.fake_img_dir = fake_img_dir
        self.real_img_list = os.listdir(real_img_dir)[:num_img+num_noise]
        self.fake_img_list = os.listdir(fake_img_dir)

        self.num_real_img = num_img + num_noise
        self.num_fake_img = num_img + num_noise
        self.num_img = num_img
        self.num_noise = num_noise // 2
        self.num_white = num_noise // 4
        self.num_black = num_noise // 4

        self.transform = transform

    def __len__(self):
        return self.num_real_img + self.num_fake_img

    def _make_random_noise_pil(self):
        # gaussian around 0.5 with std 0.2 clipped to [0,1]
        arr = np.random.normal(loc=0.5, scale=0.2, size=(64,64,3))
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def _make_white_pil(self):
        return Image.new('RGB', (64,64), (255, 255, 255))

    def _make_black_pil(self):
        return Image.new('RGB', (64,64), (0, 0, 0))

    def __getitem__(self, idx):
        if idx < self.num_real_img:
            img_path = os.path.join(self.real_img_dir, self.real_img_list[idx])
            img = Image.open(img_path).convert('RGB')
            label = 1 # real
        elif idx<self.num_real_img + self.num_img:
            img_path = os.path.join(self.fake_img_dir, self.fake_img_list[idx - self.num_real_img])
            img = Image.open(img_path).convert('RGB')
            label = 0 # fake
        else:
            if idx < self.num_real_img + self.num_img + self.num_noise:
                img = self._make_random_noise_pil()
            elif idx < self.num_real_img + self.num_img + self.num_noise + self.num_white:
                img = self._make_white_pil()
            else:
                img = self._make_black_pil()
            label = 0

        if self.transform:
            img = self.transform(img)

        return img, label