import torchvision
from torchvision import transforms
import torch
from FCN_multiclass.sp_utils.Loader_dataset import DatasetPlane, Dataset_Multiclass_heatmaps
import os
import numpy as np
from scipy.signal import butter, filtfilt
from PIL import Image

import math
import random

####_______Our functions____######
def load_split_train_val(dataroot, train_dir, val_dir, config):

    train_data = DatasetPlane(os.path.join(dataroot, train_dir), image_size= config.TEST.input_im_size, label_size = config.TEST.heatmap_size,enable_transform = config.TRAIN.Augmentation,norm=True)
    val_data = DatasetPlane(os.path.join(dataroot, val_dir), image_size= config.TEST.input_im_size, label_size = config.TEST.heatmap_size,enable_transform = False,norm=True)


    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.TRAIN.BATCH_SIZE, num_workers=0)

    valloader = torch.utils.data.DataLoader(val_data, batch_size=config.TRAIN.VAL_BATCH_SIZE)

    return trainloader, valloader


def multiclass_heatmap_load_train_val(dataroot, train_dir, val_dir, config):
    class_num = len(os.listdir(os.path.join(dataroot, train_dir)))

    train_data = Dataset_Multiclass_heatmaps(os.path.join(dataroot, train_dir), image_size=config.TEST.input_im_size,
                              label_size=config.TEST.heatmap_size, enable_transform=config.TRAIN.Augmentation,
                              norm=True)
    val_data = Dataset_Multiclass_heatmaps(os.path.join(dataroot, val_dir), image_size=config.TEST.input_im_size,
                            label_size=config.TEST.heatmap_size, enable_transform=False, norm=True)


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
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False,num_workers=0, sampler = weighted_sampler)

    valloader = torch.utils.data.DataLoader(val_data, batch_size=config.TRAIN.VAL_BATCH_SIZE)


    return trainloader, valloader, class_num


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


def load_test_data(dataroot, val_dir,config):

    val_data = DatasetPlane(os.path.join(dataroot, val_dir), image_size= config.TEST.input_im_size, label_size = config.TEST.heatmap_size,enable_transform = config.TEST.enable_transform, norm=config.TEST.normalization)

    valloader = torch.utils.data.DataLoader(val_data, batch_size=config.TEST.BATCH_SIZE)

    return valloader

def trans_norm(input_data,resize_size):
    '''Input is Image, output is tensor'''
    normalization = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformation = torchvision.transforms.Compose([torchvision.transforms.Resize((resize_size, resize_size)),

                                            torchvision.transforms.ToTensor()])

    input_data = transformation(input_data)
    input_data = normalization(input_data)
    return  input_data

def real_distance_error_calculation(dists,config):
    '''Calculate the real distance in mm for the distance error between label and heatmap
    take into account the real image size to the probe phisical dimentions (surface of contact)

    dists is between 1 and 10 form the heatmap size '''
    denorm_dists = dists *(config.TEST.heatmap_size/10)
    dist_to_input_im_size_px = (denorm_dists * (config.TEST.ORIGINAL_IMAGE_SIZE / config.TEST.heatmap_size))
    real_distance = (config.TEST.PROBE_SIZE / config.TEST.ORIGINAL_IMAGE_SIZE) * dist_to_input_im_size_px
    # (48mm/480 pix)*dist - 48mm - aperture of probe/ to pixel dist error

    return real_distance

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

def smooth(signal, N=30):
    N = 2  # Filter order
    Wn = 0.05  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    # Second, apply the filter
    filt = filtfilt(B, A, signal)
    return filt



#######_____FUNCTIONS FROM https://arxiv.org/pdf/1804.06208.pdf code - https://github.com/Microsoft/human-pose-estimation.pytorch  ###

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


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    # print(preds)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    # print(dists.shape)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                # print('normed',normed_preds,normalize[n])
                normed_targets = target[n, c, :] / normalize[n]
                # print('normed', normed_targets,normalize[n])
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                # print("just distance, not a linalg.norm ",normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        #norm is [[5.6 5.6]]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        # print("norm",norm, norm.shape)
        # print("pred,target", pred,target, pred.shape)
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]],thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred,target,dists


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


def confusion_matrix(num_classes, test_dir, model):
    transformation = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    if num_classes == 3:

        gap_list = os.listdir(os.path.join(test_dir, "Gap"))
        non_gap_list = os.listdir(os.path.join(test_dir, "NonGap"))

        gap_list = [os.path.join(test_dir, "Gap", item) for item in gap_list]
        non_gap_list = [os.path.join(test_dir, "NonGap", item) for item in non_gap_list]

        sacrum_list = os.listdir(os.path.join(test_dir, "Sacrum"))
        sacrum_list = [os.path.join(test_dir, "Sacrum", item) for item in sacrum_list]

        n_correct = 0
        n_total = 0
        n_gap_gap = 0
        n_gap_sp = 0
        n_gap_sacrum = 0
        n_sp_sp = 0
        n_sp_gap = 0
        n_sp_sacrum = 0
        n_sacrum_sacrum = 0
        n_sacrum_gap = 0
        n_sacrum_sp = 0

        for i, test_db in enumerate([gap_list, non_gap_list,sacrum_list]):
            for data in test_db:
                input_data = Image.open(data).convert(mode="RGB")
                if input_data is None:
                    return input_data
                tensor_image = transformation(input_data).unsqueeze_(0)
                image_batch, _ = list2batch([tensor_image], None)

                out = model.run_inference(image_batch)
                prob = torch.sigmoid(out).numpy()
                # print(prob)
                # print('prob',prob)
                # print('i',i)

                prob_vertebra = prob[0, 1]
                prob_gap = prob[0, 0]
                prob_sacrum = prob[0, 2]
                # print(prob_gap)
                # print(prob)

                # print('prob vert',prob_vertebra)
                # if round(prob_vertebra) == i:
                #     # n_correct += 1
                #     # print("correct")

                ###truepositive
                if round(prob_vertebra)== 1 and i == 1:
                    n_sp_sp +=1

                ###truenegative
                if round(prob_gap)== 1 and i == 0:
                    n_gap_gap +=1

                #sacrum
                if round(prob_sacrum)== 1 and i == 2:
                    n_sacrum_sacrum +=1

                #true spinous, false gap
                if round(prob_gap)== 1 and i == 1:
                    n_sp_gap +=1

                # true sacrum, false gap
                if round(prob_gap) == 1 and i == 2:
                    n_sacrum_gap += 1

                #true gap, false spinous
                if round(prob_vertebra)== 1 and i == 0:
                    n_gap_sp+=1

                # true sacrum, false spinous
                if round(prob_vertebra) == 1 and i == 2:
                    n_sacrum_sp+= 1

                # true gap, false sacrum
                if round(prob_sacrum) == 1 and i == 0:
                    n_gap_sacrum+= 1

                # true spinous, false sacrum
                if round(prob_sacrum) == 1 and i == 1:
                    n_sp_sacrum += 1


                n_total += 1

        n_correct = n_gap_gap+n_sp_sp+n_sacrum_sacrum

        # if polyaxon == "True":
        #     logger.info('correct {} ({})'.format(n_correct, n_correct/n_total))
        #     logger.info("true gap predicted spinous {} ({})".format( n_gap_sp,n_gap_sp/n_total))
        #     logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
        #     logger.info("true spinous predicted gap {} ({})".format( n_sp_gap, n_sp_gap / n_total))
        #     logger.info("true spinous predicted sacrum {} ({})".format(n_sp_sacrum, n_sp_sacrum / n_total))
        #     logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
        #     logger.info("true sacrum predicted spinous {} ({})".format(n_sacrum_sp, n_sacrum_sp / n_total))
        #     logger.info('total {}'.format(n_total))
        # else:
        print('correct',n_correct, n_correct/n_total)
        print("true gap predicted spinous", n_gap_sp,n_gap_sp/n_total)
        print("true gap predicted sacrum", n_gap_sacrum, n_gap_sacrum / n_total)
        print("true spinous predicted gap", n_sp_gap, n_sp_gap / n_total)
        print("true spinous predicted sacrum", n_sp_sacrum, n_sp_sacrum / n_total)
        print("true sacrum predicted gap", n_sacrum_gap, n_sacrum_gap / n_total)
        print("true sacrum predicted spinous", n_sacrum_sp, n_sacrum_sp / n_total)
        print('total',n_total)

    if num_classes == 4:
        gap_list = os.listdir(os.path.join(test_dir, "Gap"))
        gap_list = [os.path.join(test_dir, "Gap", item) for item in gap_list]

        # non_gap_list = os.listdir(os.path.join(test_dir, "NonGap"))
        # non_gap_list = [os.path.join(test_dir, "NonGap", item) for item in non_gap_list]

        sacrum_list = os.listdir(os.path.join(test_dir, "Sacrum"))
        sacrum_list = [os.path.join(test_dir, "Sacrum", item) for item in sacrum_list]

        lumbar_list = os.listdir(os.path.join(test_dir, "Lumbar"))
        lumbar_list = [os.path.join(test_dir, "Lumbar", item) for item in lumbar_list]

        thoracic_list = os.listdir(os.path.join(test_dir, "Thoracic"))
        thoracic_list = [os.path.join(test_dir, "Thoracic", item) for item in thoracic_list]


        n_correct = 0
        n_total = 0
        n_gap_gap = 0
        n_gap_sacrum = 0
        n_gap_lumbar = 0
        n_gap_thoracic = 0

        n_sacrum_sacrum = 0
        n_sacrum_gap = 0
        n_sacrum_lumbar = 0
        n_sacrum_thoracic = 0

        n_lumbar_lumbar = 0
        n_lumbar_gap = 0
        n_lumbar_thoracic = 0
        n_lumbar_sacrum = 0

        n_thoracic_thoracic = 0
        n_thoracic_gap = 0
        n_thoracic_sacrum = 0
        n_thoracic_lumbar = 0

        for i, test_db in enumerate([gap_list, lumbar_list, sacrum_list, thoracic_list]):
            for data in test_db:
                input_data = Image.open(data).convert(mode="RGB")
                if input_data is None:
                    return input_data
                tensor_image = transformation(input_data).unsqueeze_(0)
                image_batch, _ = list2batch([tensor_image], None)

                out = model.run_inference(image_batch)
                prob = torch.sigmoid(out).numpy()
                # print(prob)
                # print('prob',prob)
                # print('i',i)

                prob_gap = prob[0, 0]
                prob_lumbar = prob[0, 1]
                prob_sacrum = prob[0, 2]
                prob_thoracic = prob[0,3]
                # print(prob_gap)
                # print(prob)

                # print('prob vert',prob_vertebra)
                # if round(prob_vertebra) == i:
                #     # n_correct += 1
                #     # print("correct")

                ###truepositive
                if round(prob_lumbar) == 1 and i == 1:
                    n_lumbar_lumbar += 1

                ###truenegative
                if round(prob_gap) == 1 and i == 0:
                    n_gap_gap += 1

                # sacrum
                if round(prob_sacrum) == 1 and i == 2:
                    n_sacrum_sacrum += 1

                if round(prob_thoracic) == 1 and i == 3:
                    n_thoracic_thoracic += 1

                # true spinous, false gap
                if round(prob_gap) == 1 and i == 1:
                    n_lumbar_gap += 1

                # true sacrum, false gap
                if round(prob_gap) == 1 and i == 2:
                    n_sacrum_gap += 1

                # true gap, false spinous
                if round(prob_lumbar) == 1 and i == 0:
                    n_gap_lumbar += 1

                # true sacrum, false spinous
                if round(prob_lumbar) == 1 and i == 2:
                    n_sacrum_lumbar += 1

                # true gap, false sacrum
                if round(prob_sacrum) == 1 and i == 0:
                    n_gap_sacrum += 1

                # true spinous, false sacrum
                if round(prob_sacrum) == 1 and i == 1:
                    n_lumbar_sacrum += 1

                if round(prob_thoracic) == 1 and i == 0:
                    n_gap_thoracic += 1

                if round(prob_thoracic) == 1 and i == 1:
                    n_lumbar_thoracic += 1

                if round(prob_thoracic) == 1 and i == 2:
                    n_sacrum_thoracic += 1

                if round(prob_gap) == 1 and i == 3:
                    n_thoracic_gap += 1

                if round(prob_lumbar) == 1 and i == 3:
                    n_thoracic_lumbar += 1

                if round(prob_sacrum) == 1 and i == 3:
                    n_thoracic_sacrum += 1

                n_total += 1

        n_correct = n_gap_gap + n_lumbar_lumbar + n_sacrum_sacrum +n_thoracic_thoracic

        # if polyaxon == "True":
        #     logger.info('correct {} ({})'.format(n_correct, n_correct / n_total))
        #     logger.info("true gap predicted lumbar {} ({})".format(n_gap_lumbar, n_gap_lumbar / n_total))
        #     logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
        #     logger.info("true gap predicted thoracic {} ({})".format(n_gap_thoracic, n_gap_thoracic / n_total))
        #
        #     logger.info("true lumbar predicted gap {} ({})".format(n_lumbar_gap, n_lumbar_gap / n_total))
        #     logger.info("true lumbar predicted sacrum {} ({})".format(n_lumbar_sacrum, n_lumbar_sacrum / n_total))
        #     logger.info("true lumbar predicted thoracic {} ({})".format(n_lumbar_thoracic, n_lumbar_thoracic / n_total))
        #
        #     logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
        #     logger.info("true sacrum predicted lumbar {} ({})".format(n_sacrum_lumbar, n_sacrum_lumbar / n_total))
        #     logger.info("true sacrum predicted thoracic {} ({})".format(n_sacrum_thoracic, n_sacrum_thoracic / n_total))
        #
        #     logger.info("true thoracic predicted gap {} ({})".format(n_thoracic_gap, n_thoracic_gap / n_total))
        #     logger.info("true thoracic predicted lumbar {} ({})".format(n_thoracic_lumbar, n_thoracic_lumbar / n_total))
        #     logger.info("true thoracic predicted sacrum {} ({})".format(n_thoracic_sacrum, n_thoracic_sacrum / n_total))
        #
        #     logger.info('total {}'.format(n_total))
        # else:
        print('correct {} ({})'.format(n_correct, n_correct / n_total))
        print("true gap predicted lumbar {} ({})".format(n_gap_lumbar, n_gap_lumbar / n_total))
        print("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
        print("true gap predicted thoracic {} ({})".format(n_gap_thoracic, n_gap_thoracic / n_total))

        print("true lumbar predicted gap {} ({})".format(n_lumbar_gap, n_lumbar_gap / n_total))
        print("true lumbar predicted sacrum {} ({})".format(n_lumbar_sacrum, n_lumbar_sacrum / n_total))
        print("true lumbar predicted thoracic {} ({})".format(n_lumbar_thoracic, n_lumbar_thoracic / n_total))

        print("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
        print("true sacrum predicted lumbar {} ({})".format(n_sacrum_lumbar, n_sacrum_lumbar / n_total))
        print("true sacrum predicted thoracic {} ({})".format(n_sacrum_thoracic, n_sacrum_thoracic / n_total))

        print("true thoracic predicted gap {} ({})".format(n_thoracic_gap, n_thoracic_gap / n_total))
        print("true thoracic predicted lumbar {} ({})".format(n_thoracic_lumbar, n_thoracic_lumbar / n_total))
        print("true thoracic predicted sacrum {} ({})".format(n_thoracic_sacrum, n_thoracic_sacrum / n_total))

        print('total', n_total)


def list2batch(tensor_list, label_list):

    if label_list is not None and len(label_list) == 0:
        label_list = None
    if label_list is not None:
        assert len(tensor_list) == len(label_list), "Number of labels files do not match number of images files"

    tensor_batch = torch.FloatTensor()
    torch.cat(tensor_list, out=tensor_batch)

    if label_list is None:
        return tensor_batch, None

    if type(label_list[0]) is torch.Tensor:
        label_batch = torch.FloatTensor()
        torch.cat(label_list, out=label_batch)

    else:
        label_batch = torch.FloatTensor(label_list)

    return tensor_batch, label_batch




#
# def get_final_preds(config, batch_heatmaps, center, scale):
#     coords, maxvals = get_max_preds(batch_heatmaps)
#
#     heatmap_height = batch_heatmaps.shape[2]
#     heatmap_width = batch_heatmaps.shape[3]
#
#     # post-processing
#     if config.TEST.POST_PROCESS:
#         for n in range(coords.shape[0]):
#             for p in range(coords.shape[1]):
#                 hm = batch_heatmaps[n][p]
#                 px = int(math.floor(coords[n][p][0] + 0.5))
#                 py = int(math.floor(coords[n][p][1] + 0.5))
#                 if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
#                     diff = np.array([hm[py][px+1] - hm[py][px-1],
#                                      hm[py+1][px]-hm[py-1][px]])
#                     coords[n][p] += np.sign(diff) * .25
#
#     preds = coords.copy()
#
#     # Transform back
#     for i in range(coords.shape[0]):
#         preds[i] = transform_preds(coords[i], center[i], scale[i],
#                                    [heatmap_width, heatmap_height])
#
#     return preds, maxvals