# clcd_dataset.py (增强版 V3: 支持指定文件名列表)
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class CLCDDataset(Dataset):
    """
    期望目录结构：
    data/CLCD/
      ├── train/
      │   ├── time1/*.png
      │   ├── time2/*.png
      │   └── label/*.png    # 二值掩膜，变化=1，非变化=0
      ├── val/
      └── test/
    """
    def __init__(self, root_dir, split="train", transform=None, filenames=None):
        """
        初始化数据集。
        Args:
            root_dir (str): 数据集根目录。
            split (str): 数据划分 ('train', 'val', 'test')。
            transform (callable, optional): 可选的变换操作。
                                          应为一个 callable，接收 (t1, t2, label) 三个 PIL Image，
                                          并返回 (t1_tensor, t2_tensor, label_tensor)。
                                          如果为 None，则只进行基本的 ToTensor 和 Resize(256)。
            filenames (list, optional): 一个包含文件名（不含扩展名）的列表，
                                       用于指定加载该 split 下的哪些样本。
                                       如果为 None，则加载该 split 下的所有 .png 文件。
        """
        super().__init__()
        self.t1_dir = os.path.join(root_dir, split, "time1")
        self.t2_dir = os.path.join(root_dir, split, "time2")
        self.lb_dir = os.path.join(root_dir, split, "label")
        
        # --- 新增: 支持指定文件名列表 ---
        if filenames is not None:
            # 验证文件是否存在
            self.fnames = []
            for fname in filenames:
                # 假设文件名不带扩展名
                full_fname = fname + ".png" # 根据你的实际文件扩展名调整
                if all(os.path.exists(os.path.join(d, full_fname)) 
                       for d in [self.t1_dir, self.t2_dir, self.lb_dir]):
                    self.fnames.append(full_fname)
                else:
                    print(f"[Dataset] Warning: File {full_fname} not found in all directories for split '{split}', skipping.")
        else:
            # 默认加载所有 .png 文件
            self.fnames = sorted([f for f in os.listdir(self.t1_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            # 简单验证一下其他目录是否也有对应文件
            lb_fnames = set(os.listdir(self.lb_dir))
            t2_fnames = set(os.listdir(self.t2_dir))
            missing_in_lb = set(self.fnames) - lb_fnames
            missing_in_t2 = set(self.fnames) - t2_fnames
            if missing_in_lb:
                print(f"[Dataset] Warning: {len(missing_in_lb)} files missing in label dir.")
            if missing_in_t2:
                print(f"[Dataset] Warning: {len(missing_in_t2)} files missing in time2 dir.")
            # 只保留三个目录都存在的文件
            self.fnames = [f for f in self.fnames if f in lb_fnames and f in t2_fnames]

        if not self.fnames:
             raise RuntimeError(f"[Dataset] No valid files found in {self.t1_dir} (and corresponding dirs) for split '{split}'.")

        print(f"[Dataset] Found {len(self.fnames)} samples in '{split}' split.")

        # --- 存储 transform ---
        self.transform = transform

        # --- 默认 transform (如果未提供外部 transform) ---
        if self.transform is None:
            print("[Dataset] Using default internal transforms (Resize(256) + ToTensor).")
            self.default_transform = T.Compose([
                T.Resize((256, 256)),  # 默认大小
                T.ToTensor(),
            ])
        else:
            print("[Dataset] Using provided external transform.")
            self.default_transform = None

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        name = self.fnames[idx]
        # 加载图像，保持为 PIL Image
        try:
            t1 = Image.open(os.path.join(self.t1_dir, name)).convert("RGB")
            t2 = Image.open(os.path.join(self.t2_dir, name)).convert("RGB")
            # 标签图像，假设是灰度图，0 和 255
            lb = Image.open(os.path.join(self.lb_dir, name)).convert("L")
        except Exception as e:
            print(f"[Dataset] Error loading sample {name}: {e}")
            raise e

        # --- 应用 transform ---
        if self.transform is not None:
            # 假设 transform 是一个 callable，接收三个 PIL Image，返回三个 Tensor
            try:
                t1_tensor, t2_tensor, lb_tensor = self.transform(t1, t2, lb)
            except Exception as e:
                print(f"[Dataset] Error applying external transform to {name}: {e}")
                raise e
        else:
            # 使用默认 transform
            t1_tensor = self.default_transform(t1)
            t2_tensor = self.default_transform(t2)
            lb_tensor = self.default_transform(lb) # ToTensor 会自动将 0/255 转为 0.0/1.0

        # --- 拼接输入和处理标签 ---
        # 拼成 6 通道输入: [B, 6, H, W]
        x = torch.cat([t1_tensor, t2_tensor], dim=0)
        # 确保标签是二值的 float tensor: [B, 1, H, W]
        # 注意：ToTensor 将 0-255 映射到 0-1，所以 > 0.5 即可
        mask = (lb_tensor > 0.5).float() 

        return x, mask, name
