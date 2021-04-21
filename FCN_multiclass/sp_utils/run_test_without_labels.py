import numpy as np
from PIL import Image
import torch
import FCN.sp_utils as utils
from FCN.sp_utils.config import config
import matplotlib.pyplot as plt
import cv2
import torchvision as tv
import os
import keyboard
import time


def run_test_without_labels_multiclass(model,testdata,patient,device, logger,conf):
    model.eval()
    # print(model)
    probability = np.zeros(0)
    X = np.zeros(0)
    Y = np.zeros(0)
    pred = np.zeros(0)
    inputs = np.zeros(0)
    time_inference = utils.AverageMeter()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file = open('LOG1.txt', 'w')
    out = cv2.VideoWriter(config.TEST.save_dir + '_original_%s.avi' % (patient), fourcc, 3.0,
                          (1280, 480))  # for two images of size 480*640
    test_dir_patient = os.path.join(testdata, patient, "Images")
    test_list = [os.path.join(test_dir_patient, item) for item in os.listdir(test_dir_patient)]

    with torch.no_grad():
        for data in test_list:
            # logger.info("data file is {}".format(data))
            time_start = time.time()
            input_data = Image.open(data)
            # print(np.array(data))
            print("Image.open reads:",np.array(input_data).shape)
            input_data= input_data.convert("RGB")
            nonnull = np.argwhere(np.asarray(input_data)!=0)
            # print(nonnull)
            file.write(str(np.asarray(input_data)))


            # print("input data", list(input_data.getdata()))
            logger.info("size of the input image {}".format(input_data.size))
            input_data= utils.trans_norm(input_data, conf.TEST.input_im_size)

            tensor_image = input_data.unsqueeze_(0)
            logger.info("tensor of the input image {}".format(tensor_image.shape))

            inputs= tensor_image.to(device)
            # print("inputs",inputs.shape, inputs)
            logps = model.forward(inputs)
            # print(logps)

            prob_tensor = logps
            p_map = np.squeeze(prob_tensor.to("cpu").numpy())
            # logger.info("probability of spinous in frame {}".format(np.amax(p_map)))

            #### Final point prediction
            pred, _ = utils.get_max_preds(logps.detach().cpu().numpy())
            # prediction of the final point in dimentions of heatmap. Transfer it to image size
            pred = pred * config.TEST.input_im_size / config.TEST.heatmap_size
            frame_probability = np.amax(p_map)
            print("frame probability", frame_probability)
            probability = np.append(probability, frame_probability)
            X = np.append(X, pred[0][0][0])
            Y = np.append(Y, config.TEST.input_im_size-pred[0][0][1])
            # logger.info("coordinates X {}, Y{}".format(pred[0][0][0],config.TEST.input_im_size-pred[0][0][1]))

            p_map = np.multiply(p_map, 255)
            p_map_image = tv.transforms.ToPILImage()(p_map)
            p_map = tv.transforms.Resize((conf.TEST.input_im_size, conf.TEST.input_im_size))(p_map_image)

            inputs = utils.img_denorm(input_data)
            inputs = tv.transforms.ToPILImage()(inputs)

            if conf.TEST.PLOT:


                plt.subplot(1, 2, 1)
                plt.imshow(inputs)
                plt.subplot(1, 2, 2)
                plt.imshow(p_map)
                plt.scatter(x=pred[0][0][0], y=pred[0][0][1], c='r', s=40)

                xs = X
                ys = Y
                zs = probability
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xs, ys, zs, c = "#1f77b4",marker="o")

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')


                plt.show()

            if config.TEST.VIDEO == True:

                save_video(out,inputs, pred, frame_probability, patient, target=None, labels=None)

            if keyboard.is_pressed('c'):
                print("time avg", time_inference.avg)
                out.release()
                file.close()
                os._exit(0)

            time_inference.update(time.time()- time_start)
        print("time avg", time_inference.avg)
        if config.TRAIN.SWEEP_TRJ_PLOT:
            plot_path(probability,X,Y,"b")



# np.set_printoptions(threshold=sys.maxsize)
def run_test_without_labels(model,testdata,patient,device, logger,conf):
    model.eval()
    # print(model)
    probability = np.zeros(0)
    X = np.zeros(0)
    Y = np.zeros(0)
    pred = np.zeros(0)
    inputs = np.zeros(0)
    time_inference = utils.AverageMeter()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file = open('LOG1.txt', 'w')
    out = cv2.VideoWriter(config.TEST.save_dir + '_original_%s.avi' % (patient), fourcc, 3.0,
                          (1280, 480))  # for two images of size 480*640

    with torch.no_grad():
        for data in testdata:
            # logger.info("data file is {}".format(data))
            time_start = time.time()
            input_data = Image.open(data)
            # print(np.array(data))
            print("Image.open reads:",np.array(input_data).shape)
            input_data= input_data.convert("RGB")
            nonnull = np.argwhere(np.asarray(input_data)!=0)
            # print(nonnull)
            file.write(str(np.asarray(input_data)))


            # print("input data", list(input_data.getdata()))
            logger.info("size of the input image {}".format(input_data.size))
            input_data= utils.trans_norm(input_data, conf.TEST.input_im_size)

            tensor_image = input_data.unsqueeze_(0)
            logger.info("tensor of the input image {}".format(tensor_image.shape))

            inputs= tensor_image.to(device)
            # print("inputs",inputs.shape, inputs)
            logps = model.forward(inputs)
            # print(logps)

            prob_tensor = logps
            p_map = np.squeeze(prob_tensor.to("cpu").numpy())
            # logger.info("probability of spinous in frame {}".format(np.amax(p_map)))

            #### Final point prediction
            pred, _ = utils.get_max_preds(logps.detach().cpu().numpy())
            # prediction of the final point in dimentions of heatmap. Transfer it to image size
            pred = pred * config.TEST.input_im_size / config.TEST.heatmap_size
            frame_probability = np.amax(p_map)
            print("frame probability", frame_probability)
            probability = np.append(probability, frame_probability)
            X = np.append(X, pred[0][0][0])
            Y = np.append(Y, config.TEST.input_im_size-pred[0][0][1])
            # logger.info("coordinates X {}, Y{}".format(pred[0][0][0],config.TEST.input_im_size-pred[0][0][1]))

            p_map = np.multiply(p_map, 255)
            p_map_image = tv.transforms.ToPILImage()(p_map)
            p_map = tv.transforms.Resize((conf.TEST.input_im_size, conf.TEST.input_im_size))(p_map_image)

            inputs = utils.img_denorm(input_data)
            inputs = tv.transforms.ToPILImage()(inputs)

            if conf.TEST.PLOT:


                plt.subplot(1, 2, 1)
                plt.imshow(inputs)
                plt.subplot(1, 2, 2)
                plt.imshow(p_map)
                plt.scatter(x=pred[0][0][0], y=pred[0][0][1], c='r', s=40)

                xs = X
                ys = Y
                zs = probability
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(xs, ys, zs, c = "#1f77b4",marker="o")

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')


                plt.show()

            if config.TEST.VIDEO == True:

                save_video(out,inputs, pred, frame_probability, patient, target=None, labels=None)

            if keyboard.is_pressed('c'):
                print("time avg", time_inference.avg)
                out.release()
                file.close()
                os._exit(0)

            time_inference.update(time.time()- time_start)
        print("time avg", time_inference.avg)
        if config.TRAIN.SWEEP_TRJ_PLOT:
            plot_path(probability,X,Y,"b")

def plot_path(probability, X, Y,color_ext,labels = None) :
    print("Plotting path")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(xlim=(0, 224), ylim=(0, len(probability)))

    path = np.zeros(0)
    index = np.zeros(0)
    zs = 0
    for i in range(0, len(X)):
        xs = X[i]
        ys = Y[i]

        if probability[i] > 0.7 and X[i] != 0:  # Spinous
            color = color_ext
            marker = "o"
            path= np.append(path,xs)
            index = np.append(index,i)
        else:  # Gap
            color = "#8c564b"
            marker = "x"

        ax.scatter(xs, zs, c=color, marker=marker)

        zs = zs + 1
    ax.plot(path, index, label="coordinate continious")
    t = np.linspace(0, zs, zs)


    # ax1.plot(probability,t_p)
    ax.set_xlabel(' X coordinate')
    ax.set_ylabel(' step')

    Dplot = False
    if Dplot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        i = 0
        zs = 0

        for i in range(0,len(X)):
            xs = X[i]
            ys = Y[i]
            zs = zs+1
            if probability[i]>0.7: #Spinous
                color = "#1f77b4"
                marker = "o"
                ax.scatter(xs, zs, ys, c=color, marker=marker)
            else: #Gap
                color = "#1f77b4"
                marker = "x"

        t = np.linspace(0, zs*10, zs)
        ax.plot(X, t ,Y, label = "coordinate continious")
        ax.legend()
            # i =+1

        ax.set_xlabel('X Label - X coordinate')
        ax.set_ylabel('Y Label - step')
        ax.set_zlabel('Z Label - Y coordinate')
    plt.show()

def save_video(out,inputs,pred,probab_frame,patient,target = None,labels = None):

    # print("Inputs size: ",inputs.size, len(inputs.size))
    # if config.TEST.enable_transform == True:
    #     inputs = utils.img_denorm(inputs)
    #     inputs = tv.transforms.ToPILImage()(inputs)
    # else:
    #     inputs = utils.img_denorm(inputs)
    #     inputs = np.squeeze(inputs)
    #     inputs = tv.transforms.ToPILImage()(inputs)
    assert len(inputs.size)==2
    inputs = np.array(inputs)

    inputs_copy = inputs.copy()
    image_predicted = inputs
    if labels is not None:
        transformed_label = np.squeeze(labels)
        transformed_label = tv.transforms.ToPILImage()(transformed_label)
        # transformed_label = transformed_label.convert("LA")
        label_im = tv.transforms.Resize((224, 224))(transformed_label)

        label_im_cv = cv2.cvtColor(np.array(label_im), cv2.COLOR_RGB2BGR)  # cv2.COLOR_RGB2BGR, cv2.COLOR_BGR2GRAY
        # Set threshold level
        label_im_cv_gray = cv2.cvtColor(label_im_cv, cv2.COLOR_BGR2GRAY)
        # Find coordinates of all pixels below threshold
        threshold_level = 127  # half of 255
        coords = np.column_stack(np.where(label_im_cv_gray > threshold_level))

        for pose in coords:
            # print(pose)
            color_intensity = label_im_cv_gray[pose[0], pose[1]]
            # print(color_intensity)
            cv2.circle(image_predicted, (pose[1], pose[0]), 0, (int(color_intensity), 0, 0), -1)
        # print("max pixel intensity",np.amax(label_im))

    # PUT TEXT ON IMAGE - probabilities
    color = (0, 255, 0)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 200)
    fontScale = 0.5
    image_predicted = cv2.putText(image_predicted, 'probability %s' % (round(probab_frame, 2)), org, font,
                                  fontScale, color, thickness, cv2.LINE_AA)
    # Only if the probability is larger than 0.7 it is counted as detected
    if probab_frame > 0.7:
        x, y = pred[0][0][0], pred[0][0][1]
        # x = int((pred[0][0][0]) * 640 / 244)
        # y = int((pred[0][0][1]) * 480 / 244)
        x_scaled, y_scaled = int(x * (config.TEST.ORIGINAL_IMAGE_SIZE / config.TEST.input_im_size)), int(
            y * ((config.TEST.ORIGINAL_IMAGE_HEIGHT / config.TEST.input_im_size)))
        # print("predicted coordinate in dimentions 244*244", x, y)
        if labels is not None:
            print("target coordinate in dimentions 244*244", target[0][0][0], target[0][0][1])
        # print("predicted coordinate in dimentions 480*640", x_scaled, y_scaled)
        image_predicted = cv2.circle(inputs, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

        # PUT smoothed Target trajectory on to the video from pre-recorded file

    image_predicted = tv.transforms.ToPILImage()(image_predicted)
    inputs_copy = tv.transforms.ToPILImage()(inputs_copy)

    image_predicted = tv.transforms.Resize((config.TEST.ORIGINAL_IMAGE_HEIGHT, config.TEST.ORIGINAL_IMAGE_SIZE))(
        image_predicted)
    inputs_copy = tv.transforms.Resize((config.TEST.ORIGINAL_IMAGE_HEIGHT, config.TEST.ORIGINAL_IMAGE_SIZE))(
        inputs_copy)

    cv2.imshow("Result", np.hstack([inputs_copy, image_predicted]))
    cv2.waitKey(1)
    result = np.hstack([inputs_copy, image_predicted])
    out.write(np.hstack([inputs_copy, image_predicted]))

    if config.TEST.PLOT_VIDEO:
        inputs_copy = cv2.cvtColor(np.array(inputs_copy), cv2.COLOR_BGR2RGB)
        image_predicted = cv2.cvtColor(np.array(image_predicted), cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 1)
        plt.imshow(inputs_copy)
        plt.title('input', y=1.02, fontsize=10)

        plt.subplot(1, 2, 2)
        plt.imshow(image_predicted)
        plt.title('Predicted + label', y=1.02, fontsize=10)

        plt.show()

    return result

def run_FCN_streamed_image(data,model,device,probability,X,Y,logger):
    model.eval()  # Super important for testing! Otherwise the result would be random
    logger.info("Setting model to eval. It is important for testing")
    # logger.info("size of the input image {}".format(input_data.size))

    image = data.convert("RGB")
    # Don't try to write cv2.out gray frames, only BGR, otherwise the output will be empty
    # print("Image.open reads:", np.array(image).shape)
    #input data <PIL.Image.Image image mode=RGB size=640x480 at 0x274838C7320>
    input_data= utils.trans_norm(image, config.TEST.input_im_size)

    tensor_image = input_data.unsqueeze_(0)
    # logger.info("tensor of the input image {}".format(tensor_image.shape))

    inputs= tensor_image.to(device)
    # print("inputs", inputs.shape, inputs)
    logps = model.forward(inputs)
    # print(logps)

    prob_tensor = logps
    p_map = np.squeeze(prob_tensor.to("cpu").numpy())
    logger.info("probability of spinous in frame {}".format(np.amax(p_map)))

    #### Final point prediction
    pred, _ = utils.get_max_preds(logps.detach().cpu().numpy())
    # prediction of the final point in dimentions of heatmap. Transfer it to image size
    pred = pred * config.TEST.input_im_size / config.TEST.heatmap_size
    frame_probability = np.amax(p_map)
    # print("frame probability", frame_probability)
    probability = np.append(probability, frame_probability)
    X = np.append(X, pred[0][0][0])
    Y = np.append(Y, config.TEST.input_im_size-pred[0][0][1])
    # logger.info("coordinates X {}, Y{}".format(pred[0][0][0],config.TEST.input_im_size-pred[0][0][1]))

    p_map = np.multiply(p_map, 255)
    p_map_image = tv.transforms.ToPILImage()(p_map)
    p_map = tv.transforms.Resize((config.TEST.input_im_size, config.TEST.input_im_size))(p_map_image)

    inputs = utils.img_denorm(input_data)
    inputs = tv.transforms.ToPILImage()(inputs)
    return inputs,pred,probability, X, Y, frame_probability