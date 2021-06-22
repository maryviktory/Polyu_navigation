import FCN_multiclass.sp_utils as utils
from CNN_spine.rospine_utils.ModelLoader_types import ModelLoader_types
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas
import os
from PIL import Image

classes_num = 4

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


    if classes_num ==3:
        c3 = prob[0, 2]
        c3_array = np.squeeze(c3.to("cpu").numpy())

        return c1_array.tolist(), c2_array.tolist(),c3_array.tolist()


    if classes_num ==4:
        c3 = prob[0, 2]
        c4 = prob[0, 3]

        c3_array = np.squeeze(c3.to("cpu").numpy())
        c4_array = np.squeeze(c4.to("cpu").numpy())

        return c1_array.tolist(), c2_array.tolist(),c3_array.tolist(), c4_array.tolist()


    if labels is not None:
        labels = float(labels.to("cpu").item())
    else:
        return c1_array.tolist(), c2_array.tolist()

    return c1_array.tolist(), c2_array.tolist(), labels


def LoadModel(model_path):

    model = ModelLoader_types(classes_num,model_path, model_type="classification")
    model.to_device("cpu")

    model.eval()
    return  model



labels_exist = False
transformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])

test_dir = "D:\spine navigation Polyu 2021\\robot_trials_output\human experiments\Ho YIN\\1"
model_path = "D:\spine navigation Polyu 2021\DATASET_polyu\models_CNN\\trials\\3_epoch11_0894_class_4_CNN.pt"
csv_path = "D:\spine navigation Polyu 2021\\robot_trials_output\human experiments\Ho YIN\\1\\HOYIN_CNN_model_3.csv"


pd_frame = pandas.DataFrame(columns=['GAP Prob',"Sacrum Prob", "Lumbar Prob","Thoracic Prob"])
model = LoadModel(model_path)

test_list = [os.path.join(test_dir, 'Images',item) for item in os.listdir(os.path.join(test_dir, 'Images'))]

tensor_list= []
p_c1, p_c2,p_c3, p_c4 = [], [],[], []
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

            if classes_num==4:
                c1, c2, c3, c4 = processing_thread(model, tensor_list, label_list=None)
                p_c1.append(c1)
                p_c2.append(c2)
                p_c3.append(c3)
                p_c4.append(c4)

            elif classes_num==3:

                c1, c2 = processing_thread(model, tensor_list, label_list=None)
                p_c1.append(c1)
                p_c2.append(c2)
                p_c3.append(c3)
                c4 = 0
            else:


                c1, c2 = processing_thread(model, tensor_list, label_list=None)
                p_c1.append(c1)
                p_c2.append(c2)
                c3 = 0
                c4 = 0


            pd_frame = pd_frame.append({'GAP Prob':c1,"Sacrum Prob":c3, "Lumbar Prob":c2,"Thoracic Prob":c4},
                                       ignore_index=True)



            # print(p_c1)
            # print(label_list)
            tensor_list, label_list = [], []
    print("Done")
    pd_frame.to_csv(os.path.join(csv_path))
    Plot =True

    # prob_gap = prob[0, 0]
    # prob_lumbar = prob[0, 1]
    # prob_sacrum = prob[0, 2]
    # prob_thoracic = prob[0, 3]

    if Plot==True:
        plt.figure()
        ax1 = plt.subplot(4, 1, 1)
        # ax1.set_title("CNN probabilities")
        plt.xlabel('Frames', fontsize=8)
        plt.ylabel("Sacrum prob.", fontsize=8)
        ax1.plot(p_c3)

        ax2 = plt.subplot(4, 1, 2)
        # ax2.set_title("Labels")
        plt.ylabel("Gap prob", fontsize=8)
        plt.xlabel('Frames', fontsize=8)
        ax2.plot(p_c1)

        ax3 = plt.subplot(4, 1, 3)
        # ax2.set_title("Labels")
        plt.ylabel("Lumbar prob.", fontsize=8)
        plt.xlabel('Frames', fontsize=8)
        ax3.plot(p_c2)

        ax4 = plt.subplot(4, 1, 4)
        # ax2.set_title("Labels")
        plt.ylabel("Thoracic prob.", fontsize=8)
        plt.xlabel('Frames', fontsize=8)
        ax4.plot(p_c4)

        plt.show()