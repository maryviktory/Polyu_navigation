import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision as tv
import random

class DatasetPlane(Dataset):

    def __init__(self, root_dir, image_size, label_size, enable_transform = True, norm=False):

        self.root_dir = root_dir
        # self.transform = transform
        self.image_list = os.listdir(os.path.join(self.root_dir, "images"))

        self.image_list = [item for item in self.image_list if ".png" in item]
        self.normalization = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm = norm
        self.image_size = image_size
        self.label_size = label_size
        self.enable_transform = enable_transform

        # self.resize = tv.transforms.Compose([tv.transforms.ToPILImage(),tv.transforms.Resize((56, 56))])


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "Images", self.image_list[idx])
        # print(img_name)
        label_name = os.path.join(self.root_dir, "Labels_heatmaps", self.image_list[idx])
        # print(img_name)

        image = Image.open(img_name)
        image = image.convert("RGB")

        label = Image.open(label_name)
        label = label.convert("RGB")



        if self.enable_transform == True:
            rotation, translation = self.custom_transform()
            scale_image_label = self.image_size/self.label_size
            # print("transformation: rotation, translation:", rotation, '__', translation)

            image = tv.transforms.functional.affine(image, rotation, (translation,0), 1, 0) #scale_image_label*translation
            label = tv.transforms.functional.affine(label, rotation, (translation, 0), 1,0)
            # print("translation {}".format(translation), "rotation {}".format(rotation))

        image = tv.transforms.Resize((self.image_size, self.image_size))(image)
        label = tv.transforms.Resize((self.label_size, self.label_size))(label)


        image = tv.transforms.ToTensor()(image)
        label = label.convert("L")
        label = tv.transforms.ToTensor()(label)


        if self.norm:
            image = self.normalization(image)
        # print(image.shape, "loader")
        return image, label

    def custom_transform(self):

        factor = random.random()
        if factor <= 0.25:  # rotation
            rotation = random.randint(-15, 15)
            translation = 0
        elif 0.25 < factor >= 0.5:
            rotation = 0
            translation =  random.randint(-20, 20) #56*56 label image
        elif 0.5 < factor >= 0.75:
            rotation = random.randint(-15, 15)
            translation = 0.1 * random.randint(1, 3)

        else:
            rotation = 0
            translation = 0


        self.rotation = rotation
        self.translation = translation

        return  self.rotation,self.translation



class Dataset_Multiclass_heatmaps(Dataset):

    def __init__(self, root_dir, image_size, label_size, enable_transform = True, norm=False):
        self.image_list = []
        self.class_id_list = []
        self.root_dir = root_dir
        print(self.root_dir)
        # self.transform = transform
        for class_id, (class_folder) in enumerate(os.listdir(self.root_dir)):
            self.im_list = os.listdir(os.path.join(self.root_dir, class_folder))
            self.im_list = [os.path.join(self.root_dir, class_folder,item) for item in self.im_list if ".png" in item]
            self.class_id_list_one_folder = [class_id for item in self.im_list if ".png" in item]
            self.image_list.extend(self.im_list)
            self.class_id_list.extend(self.class_id_list_one_folder)
        # print(self.image_list)
        # print(self.class_id_list)

        self.normalization = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm = norm
        self.image_size = image_size
        self.label_size = label_size
        self.enable_transform = enable_transform

        # self.resize = tv.transforms.Compose([tv.transforms.ToPILImage(),tv.transforms.Resize((56, 56))])


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_list[idx]
        # print("split",os.path.split(self.root_dir))
        label_name = os.path.join(os.path.split(self.root_dir)[0], "Labels_heatmaps",  os.path.split(img_name)[-1])
        class_id = self.class_id_list[idx]
        # print(img_name)
        # print(label_name)

        image = Image.open(img_name)
        image = image.convert("RGB")

        label = Image.open(label_name)
        label = label.convert("RGB")


        if self.enable_transform == True:
            rotation, translation = self.custom_transform()
            scale_image_label = self.image_size/self.label_size
            # print("transformation: rotation, translation:", rotation, '__', translation)

            image = tv.transforms.functional.affine(image, rotation, (translation,0), 1, 0) #scale_image_label*translation
            label = tv.transforms.functional.affine(label, rotation, (translation, 0), 1,0)
            # print("translation {}".format(translation), "rotation {}".format(rotation))

        image = tv.transforms.Resize((self.image_size, self.image_size))(image)
        label = tv.transforms.Resize((self.label_size, self.label_size))(label)


        image = tv.transforms.ToTensor()(image)
        label = label.convert("L")
        label = tv.transforms.ToTensor()(label)

        if self.norm:
            image = self.normalization(image)
        # print(image.shape, "loader")

        return image, label, class_id

    def custom_transform(self):

        factor = random.random()
        if factor <= 0.25:  # rotation
            rotation = random.randint(-15, 15)
            translation = 0
        elif 0.25 < factor >= 0.5:
            rotation = 0
            translation =  random.randint(-20, 20) #56*56 label image
        elif 0.5 < factor >= 0.75:
            rotation = random.randint(-15, 15)
            translation = 0.1 * random.randint(1, 3)

        else:
            rotation = 0
            translation = 0


        self.rotation = rotation
        self.translation = translation

        return  self.rotation,self.translation