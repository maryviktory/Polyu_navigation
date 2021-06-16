import rospine_utils as utils
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import os
from PIL import Image

def processing_thread(model, input_list, label_list):
    inputs, labels = utils.list2batch(input_list, label_list)
    output = model.run_inference(inputs.to("cpu"))

    # with torch.no_grad():
    #     output = model.forward(inputs)

    prob = torch.sigmoid(output)

    c1 = prob[0, 0]
    c2 = prob[0, 1]

    c1_array = np.squeeze(c1.to("cpu").numpy())
    c2_array = np.squeeze(c2.to("cpu").numpy())

    if labels is not None:
        labels = float(labels.to("cpu").item())
    else:
        return c1_array.tolist(), c2_array.tolist()

    return c1_array.tolist(), c2_array.tolist(), labels


def LoadModel(model_path):

    model = utils.ModelLoader_types(model_path, model_type="classification")
    model.to_device("cpu")

    model.eval()
    return  model



labels_exist = True
transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

test_dir = "/media/maryviktory/My Passport/IROS 2020 TUM/Video/images_MariaT_full"
model_path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/models/spinous_best_18_retrain.pt"
csv_path = "/media/maryviktory/My Passport/IROS 2020 TUM/Video/MariaT_F10.csv"


pd_frame = pandas.DataFrame(columns=['Spinous Probability'])
model = LoadModel(model_path)

test_list = [os.path.join(test_dir, 'Images',item) for item in os.listdir(test_dir)]

tensor_list= []
p_c1, p_c2 = [], []
buffer_len = 1

if labels_exist:
    labels_list = [os.path.join(test_dir, 'Labels', item) for item in os.listdir(test_dir)]
    label_lst = []
    for current_image, current_label in test_list,labels_list:
        current_image = Image.open(current_image)
        current_image = current_image.convert(mode='RGB')
        tensor_list.append(transformation(current_image).unsqueeze_(0))
        label_lst.append(current_label)

        if len(tensor_list) >= buffer_len:
            c1, c2 = processing_thread(model, tensor_list, label_list)
            p_c1.append(c1)
            p_c2.append(c2)


            pd_frame = pd_frame.append({'Spinous Probability': c1},
                                       ignore_index=True)


            pd_frame = pd_frame.append({'Spinous Probability': c1, 'Label': labels_list},
                                       ignore_index=True)

            pd_frame.to_csv(os.path.join(csv_path))


            # print(p_c1)
            # print(label_list)
            tensor_list, label_list = [], []

else:
    for current_image in test_list:
        current_image = Image.open(current_image)
        current_image = current_image.convert(mode='RGB')
        tensor_list.append(transformation(current_image).unsqueeze_(0))

        if len(tensor_list) >= buffer_len:
            c1, c2 = processing_thread(model, tensor_list, label_list=None)
            p_c1.append(c1)
            p_c2.append(c2)

            pd_frame = pd_frame.append({'Spinous Probability': c1},
                                       ignore_index=True)

            pd_frame.to_csv(os.path.join(csv_path))

            # print(p_c1)
            # print(label_list)
            tensor_list, label_list = [], []