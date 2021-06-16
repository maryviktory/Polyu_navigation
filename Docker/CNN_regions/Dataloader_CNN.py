
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision as tv
import random

class Dataset_CNN_balanced_class(Dataset):

    def __init__(self, root_dir, config,enable_transform = True, norm=False):
        self.image_list = []
        self.class_id_list = []
        self.root_dir = root_dir
        self.config = config
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
        print("classes", os.listdir(self.root_dir))
        self.normalization = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm = norm
        self.image_size = 256
        # self.label_size = label_size
        self.enable_transform = enable_transform

        # self.resize = tv.transforms.Compose([tv.transforms.ToPILImage(),tv.transforms.Resize((56, 56))])


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_list[idx]
        # print("split",os.path.split(self.root_dir))

        class_id = self.class_id_list[idx]
        # print(img_name)
        # print(label_name)

        image = Image.open(img_name)
        image = image.convert("RGB")

        if self.enable_transform == True:
            rotation, translation = self.custom_transform()

            # print("transformation: rotation, translation:", rotation, '__', translation)

            image = tv.transforms.functional.affine(image, rotation, (translation,0), 1, 0) #scale_image_label*translation
            # label = tv.transforms.functional.affine(label, rotation, (translation, 0), 1,0)


            # print("translation {}".format(translation), "rotation {}".format(rotation))

        # Plot = False
        # if Plot == True:
        #     print(img_name)
        #     print("class id", class_id)
        #     plt.figure()
        #     plt.imshow(np.asarray(image))
        #     plt.figure()
        #     plt.imshow(np.asarray(label))
        #     plt.show()

        image = tv.transforms.Resize((self.image_size, self.image_size))(image)
        # label = tv.transforms.Resize((self.label_size, self.label_size))(label)


        image = tv.transforms.ToTensor()(image)


        if self.norm:
            image = self.normalization(image)
        # print(image.shape, "loader")

        return image, class_id

    def custom_transform(self):

        factor = random.random()
        if factor <= 0.25:  # rotation
            rotation = random.randint(-30, 30)
            translation = 0
        elif 0.25 < factor >= 0.5:
            rotation = 0
            translation =  random.randint(-20, 20) #56*56 label image
        elif 0.5 < factor >= 0.75:
            rotation = random.randint(-30, 30)
            translation = 0.1 * random.randint(1, 3)

        else:
            rotation = 0
            translation = 0

        self.rotation = rotation
        self.translation = translation

        return  self.rotation,self.translation


def make_weights_for_balanced_classes(images, nclasses):

    classes_list = os.listdir(images)
    weight = [0] * nclasses
    for i, class_name in enumerate(classes_list):
        list_class = [index for index in os.listdir(os.path.join(images,class_name))]
        weight[i] = len(list_class)
    # print(weight)
    # weight = torch.Tensor([0.0008, 0.0008, 0.0007, 0.0007])
    weight = 1 / torch.Tensor(weight)
    # sum = len(list_class[0]+list_class[1]+list_class[2]+list_class[3])

    # weight = np.array(weight)//sum


    # for item in enumerate(os.listdir(images)):
    #     count[item[1]] += 1
    # weight_per_class = [0.] * nclasses
    # N = float(sum(count))
    # for i in range(nclasses):
    #     weight_per_class[i] = N/float(count[i])
    # weight = [0] * len(images)
    # for idx, val in enumerate(images):
    #     weight[idx] = weight_per_class[val[1]]
    return weight





def CNN_load_train_val(dataroot, config):
    train_dir = "train"
    val_dir = "val"
    class_num = len(os.listdir(os.path.join(dataroot, train_dir)))

    train_data = Dataset_CNN_balanced_class(os.path.join(dataroot, train_dir),
                               config=config, enable_transform=config.Augmentation,
                              norm=True)
    val_data = Dataset_CNN_balanced_class(os.path.join(dataroot, val_dir),
                            config=config, enable_transform=False, norm=True)


    target_list = torch.tensor(train_data.class_id_list)
    # target_list = target_list[torch.randperm(len(target_list))]

    weights_classes = make_weights_for_balanced_classes(os.path.join(dataroot, train_dir), class_num)
    print("weights_classes",weights_classes)

    class_weights_all = weights_classes[target_list]
    # print(class_weights_all)

    weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=False,num_workers=0, sampler = weighted_sampler)

    valloader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE)


    return trainloader, valloader, class_num