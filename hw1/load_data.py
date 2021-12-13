import numpy as np
import os
import random
import torch


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    import imageio
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}, device = torch.device('cpu')):
        """
        Args:
            num_classes: int
                Number of classes for classification (N-way)
            
            num_samples_per_class: int
                Number of samples per class in the support set (K-shot).
                Will generate additional sample for the query set.
                
            device: cuda.device: 
                Device to allocate tensors to.
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]
        self.device = device

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: str
                train/val/test set to sample from
                
            batch_size: int:
                Size of batch of tasks to sample
                
        Returns:
            images: tensor
                A tensor of images of size [B, K+1, N, 784]
                where B is batch size, K is number of samples per class, 
                N is number of classes
                
            labels: tensor
                A tensor of images of size [B, K+1, N, N] 
                where B is batch size, K is number of samples per class, 
                N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        #############################
        K = self.num_samples_per_class
        B = batch_size
        N = self.num_classes
        sampled_classes = np.random.permutation(range(len(folders)))[:N]
        sampled_folders = [folders[i] for i in sampled_classes]
        images = []
        for i in range(B):
            batched_files  = get_images(sampled_folders, range(N), K+1, shuffle=False)
            batched_images = [image_file_to_array(sample[1], self.dim_input) for sample in batched_files]    
            batched_images = np.stack(batched_images, 0).reshape((N, K+1, self.dim_input)).transpose((1, 0, 2))
            images.append(batched_images)
        images = np.stack(images, 0)
        labels = np.eye(N, dtype=int)[None,None,:,:]
        labels = np.repeat(labels, B,   axis=0)
        labels = np.repeat(labels, K+1, axis=1)

        val_images   = images[:,-1,:,:].copy()
        val_labels   = labels[:,-1,:,:].copy()

        for i in range(B):
            inds = np.random.permutation(range(N))
            labels[i, -1] = val_labels[i, inds]
            images[i, -1] = val_images[i, inds]

        return torch.from_numpy(images).to(self.device), torch.from_numpy(labels).to(self.device) 

        # K = self.num_samples_per_class
        # B = batch_size
        # N = self.num_classes
        # sampled_classes = np.random.permutation(range(len(folders)))[:N]
        # sampled_folders = [folders[i] for i in sampled_classes]
        # images, labels = [], []
        # for i in range(B):
        #     train_files  = get_images(sampled_folders, range(N), K, shuffle=False)
        #     val_files    = get_images(sampled_folders, range(N), 1, shuffle=True)
        #     train_images = [image_file_to_array(sample[1], self.dim_input) for sample in train_files]
        #     val_images   = [image_file_to_array(sample[1], self.dim_input) for sample in val_files]
        #     train_labels, val_labels = [], []
        #     for sample in train_files:
        #         onehot = np.zeros(N)
        #         onehot[sample[0]] = 1
        #         train_labels.append(onehot)
        #     for sample in val_files:
        #         onehot = np.zeros(N)
        #         onehot[sample[0]] = 1
        #         val_labels.append(onehot)
        #     train_labels = np.stack(train_labels, 0).reshape((N, K, N)).transpose((1, 0, 2))
        #     val_labels   = np.stack(val_labels, 0).reshape((1, N, N))
        #     train_images = np.stack(train_images, 0).reshape((N, K, self.dim_input)).transpose((1, 0, 2))
        #     val_images   = np.stack(val_images, 0).reshape((1, N, self.dim_input))
        #     batched_images = torch.cat((train_images, val_images), 0)
        #     batched_labels = torch.cat((train_labels, val_labels), 0)
        #     images.append(batched_images)
        #     labels.append(batched_labels)
        # images = np.stack(images, 0)
        # labels = np.stack(labels)        
        # return torch.from_numpy(images).to(self.device), torch.from_numpy(labels).to(self.device) 