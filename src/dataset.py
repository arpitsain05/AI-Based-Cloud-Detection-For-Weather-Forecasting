import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from src.config import CLASSES, IMAGE_SIZE, NORM_MEAN, NORM_STD, BATCH_SIZE, NUM_WORKERS
except ImportError:
    from config import CLASSES, IMAGE_SIZE, NORM_MEAN, NORM_STD, BATCH_SIZE, NUM_WORKERS


class SatelliteWeatherDataset(Dataset):
    """
    Custom PyTorch Dataset for loading satellite weather images.
    Supports reading from directory layouts of the form:
    data/processed/<split>/<class_name>/<image_name>
    """
    def __init__(self, data_dir, classes=CLASSES, transform=None):
        self.data_dir = data_dir
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Create a mapping from class name to class index
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Scans the directory structure and populates paths and labels."""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory '{self.data_dir}' does not exist yet.")
            return

        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                # We skip missing subdirectories (e.g. if we are doing a test run)
                continue
                
            class_idx = self.class_to_idx[class_name]
            # List image files in the class subdirectory
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(class_path, file_name))
                    self.labels.append(class_idx)
                    
        print(f"Loaded {len(self.image_paths)} images from {self.data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image with OpenCV (BGR format) and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply Albumentations augmentations/transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label


def get_transforms(img_size=IMAGE_SIZE, mean=NORM_MEAN, std=NORM_STD):
    """
    Defines Albumentations transform pipelines for train, validation and test splits.
    """
    train_transform = A.Compose([
        # Resize image to the expected backbone size
        A.Resize(height=img_size, width=img_size),
        
        # Spatial augmentations - Satellite views have no native up/down orientation
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        
        # Pixel-level augmentations to simulate sun glint and haze variations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        
        # Standard Normalization & conversion to Tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    val_test_transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, val_test_transform


def get_dataloaders(
    train_dir, val_dir, test_dir, 
    batch_size=BATCH_SIZE, 
    img_size=IMAGE_SIZE, 
    num_workers=NUM_WORKERS
):
    """
    Constructs PyTorch DataLoaders for all three splits.
    """
    train_tr, val_test_tr = get_transforms(img_size=img_size)
    
    train_dataset = SatelliteWeatherDataset(train_dir, transform=train_tr)
    val_dataset = SatelliteWeatherDataset(val_dir, transform=val_test_tr)
    test_dataset = SatelliteWeatherDataset(test_dir, transform=val_test_tr)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # --- pipeline self-test ---
    print("Executing dataset pipeline verification...")
    import tempfile
    
    # We will create a temporary directory structures to test loading
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating a temporary mock directory structure at: {tmpdir}")
        for cls in CLASSES:
            cls_dir = os.path.join(tmpdir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            # Create a mock 100x100 random noise RGB image
            mock_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            # Save 2 mock images per class
            for i in range(2):
                cv2.imwrite(os.path.join(cls_dir, f"mock_img_{i}.jpg"), mock_img)
                
        # Initialize Dataset
        train_tr, _ = get_transforms()
        dataset = SatelliteWeatherDataset(tmpdir, transform=train_tr)
        
        print(f"Dataset length verified: {len(dataset)} (Expected: {2 * len(CLASSES)})")
        assert len(dataset) == 2 * len(CLASSES), "Dataset length does not match mock count."
        
        # Retrieve a sample
        image, label = dataset[0]
        print(f"Sample retrieved. Image tensor shape: {image.shape}, Label class index: {label}")
        print(f"Image tensor min/max values: {image.min().item():.4f} / {image.max().item():.4f}")
        
        # Initialize DataLoader (using num_workers=0 to prevent multiprocessing overhead in tests)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch_images, batch_labels = next(iter(loader))
        
        print(f"Batch loaded successfully!")
        print(f"Batch images tensor shape: {batch_images.shape} (Expected: [4, 3, 224, 224])")
        print(f"Batch labels tensor shape: {batch_labels.shape} (Expected: [4])")
        
        assert batch_images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE), "Image batch shape mismatch."
        assert batch_labels.shape == (4,), "Label batch shape mismatch."
        
        print("\nDataset and transformation pipeline successfully validated! [SUCCESS]")
