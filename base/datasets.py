import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random


# MNIST
from torchvision.datasets import MNIST


class RegressionMNIST(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        """
        Initialize the RegressionMNIST dataset with the specified transformations
        and settings.

        Parameters:
        - root (str): Directory where the MNIST data is stored or will be downloaded.
        - train (bool): If True, creates dataset from training set, otherwise from test set.
        - download (bool): If True, downloads the data from the internet if it's not available at root.
        - transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        """
        self.mnist = MNIST(root=root, train=train, download=download, transform=None)  # Load data without initial transform
        if transform is not None:
            self.transform = transform  # Image transformation
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensor
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
            ])

    def __getitem__(self, index):
        """
        Retrieve an image and its label, apply transformations, and adjust the label format.

        Parameters:
        - index (int): Index of the item in the dataset

        Returns:
        - tuple: Tuple (image, target), where target is the label transformed to a 1D float tensor.
        """
        img, target = self.mnist[index]

        # Apply the transformation to the image
        if self.transform:
            img = self.transform(img)

        # Transform the target to a 1D float tensor
        target = torch.tensor([target], dtype=torch.float)

        return img, target

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.mnist)


class PairMNIST(Dataset):
    def __init__(self, op_fun, root, train=True, transform=None, download=False, seed=42):
        self.op_fun = op_fun

        if transform is not None:
            self.transform = transform  # Image transformation
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Convert images to tensor
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize the dataset
            ])
            
        self.dataset = RegressionMNIST(root=root, train=train, transform=self.transform, download=download)
        self.transform = transform
        self.random_seed = seed
        random.seed(self.random_seed)  # Set the seed

        # Create a list of random indices for the second image in each pair
        self.indices = [random.randint(0, len(self.dataset) - 1) for _ in range(len(self.dataset))]

    def __getitem__(self, index):
        X1, y1 = self.dataset[index]
        X2, y2 = self.dataset[self.indices[index]]         
        X = torch.stack([X1, X2])
        y = self.op_fun(torch.concat((y1, y2)).unsqueeze(0)).squeeze(0)
        return X, y
    
    def __len__(self):
        return len(self.dataset)



class MathematicalFunction1:
    def __init__(self, low=0, high=9):
        """
        Initialize the MathematicalFunction1 class with a specified range for generating random data.
        
        Parameters:
        - low (float): The lower bound of the uniform distribution.
        - high (float): The upper bound of the uniform distribution.
        """
        self.low = low
        self.high = high

    def generate_dataset(self, num_samples):
        """
        Generates random data and calculates output using the mathematical function.
        
        Parameters:
        - num_samples (int): Number of samples to generate.
        
        Returns:
        - tuple(torch.Tensor): A tuple containing the input data X and the corresponding output y.
        """
        X = torch.rand(num_samples, 2) * (self.high - self.low) + self.low
        y = self.apply(X)
        return TensorDataset(X,y)

    @staticmethod
    def apply(X):
        """
        Applies a complex mathematical function combining multiplication, sine, cosine, and square operations to the input data.
        
        Parameters:
        - X (torch.Tensor): Input tensor of shape (num_samples, 2).
        
        Returns:
        - torch.Tensor: Output tensor obtained by calculating the mathematical function on X.
        """
        flat_y = (X[:, 0] * X[:, 1]) + torch.sin(X[:, 0]) - torch.cos(X[:, 1]) + (X[:, 0] ** 2) - (X[:, 1] ** 2)
        return flat_y.view(-1,1)


