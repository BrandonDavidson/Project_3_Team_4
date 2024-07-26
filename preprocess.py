import os
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split

class ImagePreprocessor:
    def __init__(self, image_dir, label_dir, img_size=(224, 224), batch_size=32, test_split=0.2):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_split = test_split
        
    def load_data(self):
        images = []
        labels = []
        
        # Load images
        for img_name in sorted(os.listdir(self.image_dir)):
            img_path = os.path.join(self.image_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            images.append(np.array(img))
        
        # Load labels
        label_file = os.path.join(self.label_dir, 'labels.txt')  # Assuming labels are in a single file
        with open(label_file, 'r') as f:
            for line in f:
                label = int(line.strip())  # Assuming labels are integers
                labels.append(label)
        
        # Ensure the number of images and labels match
        assert len(images) == len(labels), "Number of images and labels don't match"
        
        # Get unique class names
        class_names = sorted(set(labels))
        
        return np.array(images), np.array(labels), class_names
    
    def preprocess_tensorflow(self):
        images, labels, class_names = self.load_data()
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=self.test_split, stratify=labels, random_state=42
        )
        
        # Define preprocessing steps
        preprocess = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.img_size[0], self.img_size[1]),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ])
        
        # Create datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = (
            train_ds
            .shuffle(1000)
            .map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = (
            test_ds
            .map(lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        return train_ds, test_ds, class_names
    
    def preprocess_pytorch(self):
        images, labels, class_names = self.load_data()
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=self.test_split, stratify=labels, random_state=42
        )
        
        # Define preprocessing steps
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train).permute(0, 3, 1, 2).float(),
            torch.tensor(y_train)
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test).permute(0, 3, 1, 2).float(),
            torch.tensor(y_test)
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, test_loader, class_names

# Usage example
preprocessor = ImagePreprocessor(
    image_dir='../data/train/images/',
    label_dir='../data/train/labels/',
    img_size=(224, 224),
    batch_size=32
)

# For TensorFlow
tf_train_ds, tf_test_ds, tf_class_names = preprocessor.preprocess_tensorflow()

# For PyTorch
# torch_train_loader, torch_test_loader, torch_class_names = preprocessor.preprocess_pytorch()
print (tf_train_ds)