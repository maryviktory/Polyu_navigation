import cv2
import time
import torch
import torchvision as tv
import os
import keyboard
import time
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from Spine_navigation_Polyu.utils.config_robot_GUI import config



class Kalman_filter_x_im():
    def __init__(self):
        # allocate space for arrays
        self.xhat = np.zeros(0)  # a posteri estimate of x
        self.P = np.zeros(0)  # a posteri error estimate
        self.xhatminus = np.zeros(0)  # a priori estimate of x
        self.Pminus = np.zeros(0)  # a priori error estimate
        self.K = np.zeros(0)  # gain or blending factor
        self.R = config.FORCE.Kalman_R # estimate of measurement variance, change to see effect
        self.Q = config.FORCE.Kalman_Q
        self.xhat = np.append(self.xhat,-config.FORCE.Fref)
        self.P = np.append(self.P,0)

    def update(self, z):
        # time update
        self.xhatminus = np.append(self.xhatminus,self.xhat[-1])  # +B*0.01
        self.Pminus = np.append(self.Pminus ,self.P[-1] + self.Q)
        # measurement update
        self.K = np.append(self.K, self.Pminus[-1] / (self.Pminus[-1] + self.R))
        # print("K[k]",K[k])
        self.xhat = np.append(self.xhat,self.xhatminus[-1] + self.K[-1] * (z - self.xhatminus[-1]))
        self.P = np.append(self.P,(1 - self.K[-1]) * self.Pminus[-1])
        return self.xhat[-1]

def find_centroid(c):
    M = cv2.moments(c)
    # calculate x,y coordinate of center
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return cX,cY

def reset_FT300(rob, wait=True):
    prg = '''def prog():
    if(socket_open("127.0.0.1",63350,"acc")): 
        socket_send_string("SET ZRO","acc")
        sleep(0.1)
        socket_close("acc")
    end
end
'''
    print("FT300 reset")
    programString = prg
    rob.send_program(programString)
    time.sleep(0.25)

def img_denorm(img):
    # for ImageNet the mean and std are:
    mean = np.asarray([ 0.485, 0.456, 0.406 ])
    std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = torchvision.transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = np.squeeze(img) #res = img.squeeze(0)
    # print(res.size)

    res = denormalize(res)

    # Image needs to be clipped since the denormalize function will map some
    # values below 0 and above 1
    res = torch.clamp(res, 0, 1)


    return (res)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def trans_norm(input_data,resize_size):
    '''Input is Image, output is tensor'''
    normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformation = torchvision.transforms.Compose([torchvision.transforms.Resize((resize_size, resize_size)),

                                            torchvision.transforms.ToTensor()])

    input_data = transformation(input_data)
    input_data = normalization(input_data)
    return  input_data

def save_video_original(out,inputs):
    try:
        assert len(inputs.size)==2
        inputs = np.array(inputs)
    except:
        inputs = np.array(inputs)
    inputs_copy = inputs.copy()
    # print(inputs_copy)
    inputs_copy = tv.transforms.ToPILImage()(inputs_copy)
    inputs_copy = tv.transforms.Resize((config.IMAGE.ORIGINAL_IMAGE_HEIGHT, config.IMAGE.ORIGINAL_IMAGE_SIZE))(
        inputs_copy)
    inputs_copy = cv2.cvtColor(np.array(inputs_copy), cv2.COLOR_BGR2RGB)
    out.write(inputs_copy)

    return inputs_copy


def save_video(out,inputs,pred,probab_frame,patient,config,target = None,labels = None):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(os.path.join(config.IMAGE.SAVE_PATH, 'output_original.avi'), fourcc, 3.0,
    #                       (1280, 480))  # for images of size 480*640

    # print("Inputs size: ",inputs.size, len(inputs.size))
    # if config.TEST.enable_transform == True:
    #     inputs = utils.img_denorm(inputs)
    #     inputs = tv.transforms.ToPILImage()(inputs)
    # else:
    #     inputs = utils.img_denorm(inputs)
    #     inputs = np.squeeze(inputs)
    #     inputs = tv.transforms.ToPILImage()(inputs)
    try:
        assert len(inputs.size)==2
        inputs = np.array(inputs)
    except:
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
    if probab_frame > 0.5:
        x, y = int(pred[0][0][0]), int(pred[0][0][1])
        # print(x,y)
        # x = int((pred[0][0][0]) * 640 / 244)
        # y = int((pred[0][0][1]) * 480 / 244)
        x_scaled, y_scaled = int(x * (config.IMAGE.ORIGINAL_IMAGE_SIZE / config.IMAGE.input_im_size)), int(
            y * ((config.IMAGE.ORIGINAL_IMAGE_HEIGHT / config.IMAGE.input_im_size)))
        # print("predicted coordinate in dimentions 244*244", x, y)
        if labels is not None:
            print("target coordinate in dimentions 244*244", target[0][0][0], target[0][0][1])
        # print("predicted coordinate in dimentions 480*640", x_scaled, y_scaled)
        image_predicted = cv2.circle(inputs, (x, y), radius=1, color=(0, 0, 255), thickness=-1)

        # PUT smoothed Target trajectory on to the video from pre-recorded file

    image_predicted = tv.transforms.ToPILImage()(image_predicted)
    inputs_copy = tv.transforms.ToPILImage()(inputs_copy)

    image_predicted = tv.transforms.Resize((config.IMAGE.ORIGINAL_IMAGE_HEIGHT, config.IMAGE.ORIGINAL_IMAGE_SIZE))(
        image_predicted)
    inputs_copy = tv.transforms.Resize((config.IMAGE.ORIGINAL_IMAGE_HEIGHT, config.IMAGE.ORIGINAL_IMAGE_SIZE))(
        inputs_copy)
    result = np.hstack([inputs_copy, image_predicted])
    # print(out)
    out.write(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    # print(result.shape)

    cv2.imshow("Result", np.hstack([inputs_copy, image_predicted]))
    cv2.waitKey(1)

    # result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
     #cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    if config.IMAGE.PLOT_VIDEO:
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

def save_video_filtered_points(out,inputs,pred,probab_frame,x_filtered,y_filtered, patient,config,target = None,labels = None):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(os.path.join(config.IMAGE.SAVE_PATH, 'output_original.avi'), fourcc, 3.0,
    #                       (1280, 480))  # for images of size 480*640

    # print("Inputs size: ",inputs.size, len(inputs.size))
    # if config.TEST.enable_transform == True:
    #     inputs = utils.img_denorm(inputs)
    #     inputs = tv.transforms.ToPILImage()(inputs)
    # else:
    #     inputs = utils.img_denorm(inputs)
    #     inputs = np.squeeze(inputs)
    #     inputs = tv.transforms.ToPILImage()(inputs)
    try:
        assert len(inputs.size)==2
        inputs = np.array(inputs)
    except:
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
    if probab_frame > config.IMAGE.probability_threshold:
        x, y = int(pred[0][0][0]), int(pred[0][0][1])
        # print(x,y)
        # x = int((pred[0][0][0]) * 640 / 244)
        # y = int((pred[0][0][1]) * 480 / 244)
        x_scaled, y_scaled = int(x * (config.IMAGE.ORIGINAL_IMAGE_SIZE / config.IMAGE.input_im_size)), int(
            y * ((config.IMAGE.ORIGINAL_IMAGE_HEIGHT / config.IMAGE.input_im_size)))
        # print("predicted coordinate in dimentions 244*244", x, y)
        if labels is not None:
            print("target coordinate in dimentions 244*244", target[0][0][0], target[0][0][1])
        # print("predicted coordinate in dimentions 480*640", x_scaled, y_scaled)
        image_predicted = cv2.circle(inputs, (x, y), radius=1, color=(0, 0, 255), thickness=-1)


    x_filtered = int(x_filtered * (config.IMAGE.input_im_size / config.IMAGE.ORIGINAL_IMAGE_SIZE))
    y_filtered = int(y_filtered * (config.IMAGE.input_im_size / config.IMAGE.ORIGINAL_IMAGE_SIZE))

    image_predicted = cv2.circle(image_predicted, (x_filtered, y_filtered), radius=1, color=(0, 255, 0),
                                 thickness=-1)
        # PUT smoothed Target trajectory on to the video from pre-recorded file

    image_predicted = tv.transforms.ToPILImage()(image_predicted)
    inputs_copy = tv.transforms.ToPILImage()(inputs_copy)

    image_predicted = tv.transforms.Resize((config.IMAGE.ORIGINAL_IMAGE_HEIGHT, config.IMAGE.ORIGINAL_IMAGE_SIZE))(
        image_predicted)



    inputs_copy = tv.transforms.Resize((config.IMAGE.ORIGINAL_IMAGE_HEIGHT, config.IMAGE.ORIGINAL_IMAGE_SIZE))(
        inputs_copy)
    result = np.hstack([inputs_copy, image_predicted])
    # print(out)
    out.write(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
    # print(result.shape)

    cv2.imshow("Result", np.hstack([inputs_copy, image_predicted]))
    cv2.waitKey(1)

    # result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
     #cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    if config.IMAGE.PLOT_VIDEO:
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








def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    # print("preds, maxval ",preds,maxvals)
    return preds, maxvals



def run_FCN_streamed_image(data,model,device,probability,X,Y,logger,config,**kwargs):

    # logger.info("size of the input image {}".format(input_data.size))

    image = data.convert("RGB")
    # Don't try to write cv2.out gray frames, only BGR, otherwise the output will be empty
    # print("Image.open reads:", np.array(image).shape)
    #input data <PIL.Image.Image image mode=RGB size=640x480 at 0x274838C7320>
    input_data= trans_norm(image, config.IMAGE.input_im_size)

    tensor_image = input_data.unsqueeze_(0)
    # logger.info("tensor of the input image {}".format(tensor_image.shape))

    inputs = tensor_image.to(device)
    # print(inputs)
    # print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
    # print("inputs", inputs.shape, inputs)
    # print("Model on cuda: ",next(model.parameters()).is_cuda)
    # print("Inputs on cuda: ", inputs.is_cuda)
    start_time = time.time()
    if config.TRAIN.three_heads == True:
        logps,logps_2,classification = model.forward(inputs)
    elif config.TRAIN.two_heads == True and config.TRAIN.two_heads_for_class_only == False:
        logps, classification = model.forward(inputs)

    elif config.TRAIN.two_heads == True and config.TRAIN.two_heads_for_class_only == True:
        _, classification = model.forward(inputs)
        model_FCN = kwargs['model2']
        logps = model_FCN.forward(inputs)
    else:
        logps = model.forward(inputs)
    # print("time:", time.time()-start_time)
    # print(logps)

    prob_tensor = logps
    p_map = np.squeeze(prob_tensor.to("cpu").numpy())
    # logger.info("probability of spinous in frame {}".format(np.amax(p_map)))

    #### Final point prediction
    pred, _ = get_max_preds(logps.detach().cpu().numpy())
    # prediction of the final point in dimentions of heatmap. Transfer it to image size
    pred = pred * config.IMAGE.input_im_size / config.IMAGE.heatmap_size
    frame_probability = np.amax(p_map)
    # print("frame probability", frame_probability)
    probability = np.append(probability, frame_probability)
    X = np.append(X, pred[0][0][0])
    Y = np.append(Y, pred[0][0][1])
    # logger.info("coordinates X {}, Y{}".format(pred[0][0][0],config.TEST.input_im_size-pred[0][0][1]))

    p_map = np.multiply(p_map, 255)
    p_map_image = tv.transforms.ToPILImage()(p_map)
    p_map = tv.transforms.Resize((config.IMAGE.input_im_size, config.IMAGE.input_im_size))(p_map_image)

    inputs = img_denorm(input_data)
    inputs = tv.transforms.ToPILImage()(inputs)

    if config.TRAIN.three_heads == True:
        p_map_2 = np.squeeze(logps_2.to("cpu").numpy())
        pred_2, _ = get_max_preds(logps_2.detach().cpu().numpy())
        # prediction of the final point in dimentions of heatmap. Transfer it to image size
        pred_2 = pred_2 * config.IMAGE.input_im_size / config.IMAGE.heatmap_size
        frame_probability_2 = np.amax(p_map_2)
        X_2 = pred_2[0][0][0]
        Y_2 = pred_2[0][0][1]

        classification = torch.sigmoid(classification).detach().cpu().numpy()
        return inputs,pred,probability, X, Y, frame_probability,pred_2,frame_probability_2,X_2,Y_2,classification

    elif config.TRAIN.two_heads == True:
        classification = torch.sigmoid(classification).detach().cpu().numpy()
        return inputs, pred, probability, X, Y, frame_probability, classification
    else:
        return inputs,pred,probability, X, Y, frame_probability


