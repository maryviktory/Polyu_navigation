import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import copy
import argparse
import os
import torch
import torch.optim as optim
import logging
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from PIL import ImageFile
from ModelLoader_types import ModelLoader_types
from Dataloader_CNN import CNN_load_train_val,Dataset_CNN_balanced_class
from config import config

import stat
from datetime import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True
# from utils_polyaxon import *

# Polyaxon





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




class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """



    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        self.paths = path
        # class_id = self.class_id_list[index]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def train_model(trainloader, valloader,model, criterion, optimizer, scheduler, save_path, num_epochs,model_name,feature_extract, use_pretrained,dataset):
    print("Model on cuda: ", next(model.parameters()).is_cuda)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    writer = SummaryWriter(save_path)
    dataloaders = {'train':trainloader, 'val':valloader}

    for epoch in range(num_epochs):
        print('epoch:',epoch)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            class_2 = 0
            class_1 = 0
            class_0 = 0
            class_3 = 0
            num_train = 0
            num_val = 0
            # Iterate over data.

            for inputs, labels in dataloaders[phase]:
                # print(inputs.shape)
                # print(labels.shape)


                for id in labels:
                    if id == 2:
                        class_2 = class_2 + 1
                    if id == 1:
                        class_1 = class_1 + 1
                    if id == 0:
                        class_0 = class_0 + 1
                    if id == 3:
                        class_3 = class_3 + 1
                # zero the parameter gradients

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == "train":
                    num_train = num_train+inputs.size()[0]
                else:
                    num_val = num_val+inputs.size()[0]
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # print(running_corrects)
                if epoch == 49:

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, os.path.join(save_path, "model" + str(epoch) + ".pt"))

            if phase == 'train':
                print("cl0", class_0)
                print("cl1", class_1)
                print("cl2", class_2)
                print("cl3", class_3)
                print("sceduler LR", scheduler.get_last_lr())
                logger.info("sceduler LR {}".format(scheduler.get_last_lr()))
                scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / num_train
                epoch_acc = running_corrects.double() / num_train
                print("num_train, should be around 26648",num_train)
            else:
                epoch_loss = running_loss / num_val
                epoch_acc = running_corrects.double() / num_val


            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train':
                writer.add_scalar('Loss/train', float(epoch_loss), epoch)
                writer.add_scalar('Accuracy/train', float(epoch_acc), epoch)
                # print(epoch_acc, 'accuracy')
                # print(epoch_loss,'loss')
            elif phase == 'val':
                writer.add_scalar('Loss/val', float(epoch_loss), epoch)
                writer.add_scalar('Accuracy/val', float(epoch_acc), epoch)
            # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, os.path.join(save_path, "model_best_%s_fixed_%s_pretrained_%s_%s.pt" %(model_name,feature_extract, use_pretrained,dataset)))

    model_save_path = os.path.join(save_path, "model_best_%s_fixed_%s_pretrained_%s_%s.pt" %(model_name,feature_extract, use_pretrained,dataset))

    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,model_save_path


def test_model(test_dir, model_path,num_classes,flag_Polyaxon):


    model = ModelLoader_types(num_classes, model_path, model_type="classification")
    model.eval()
    transformation = transforms.Compose([transforms.Resize(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

    if num_classes == 2:
        gap_list = os.listdir(os.path.join(test_dir, "Gap"))
        non_gap_list = os.listdir(os.path.join(test_dir, "NonGap"))

        gap_list = [os.path.join(test_dir, "Gap", item) for item in gap_list]
        non_gap_list = [os.path.join(test_dir, "NonGap", item) for item in non_gap_list]

        n_correct = 0
        n_total = 0
        n_tp = 0
        n_fn = 0
        n_fp = 0
        n_tn = 0

        for i, test_db in enumerate([gap_list, non_gap_list]):
            for data in test_db:
                input_data = Image.open(data).convert(mode="RGB")
                if input_data is None:
                    return input_data
                tensor_image = transformation(input_data).unsqueeze_(0)
                image_batch, _ = list2batch([tensor_image], None)

                out = model.run_inference(image_batch)
                prob = torch.sigmoid(out).numpy()
                # print('prob',prob)
                # print('i',i)

                prob_vertebra = prob[0, 1]
                prob_gap = prob[0, 0]
                # print(prob_gap)
                # print(prob)

                # print('prob vert',prob_vertebra)
                if round(prob_vertebra) == i:
                    n_correct += 1

                ###truepositive
                if round(prob_vertebra) == 1 and i == 1:
                    n_tp += 1

                ###truenegative
                if round(prob_vertebra) == 0 and i == 0:
                    n_tn += 1

                # false negative
                if round(prob_vertebra) == 0 and i == 1:
                    n_fn += 1

                ## false positive
                if round(prob_vertebra) == 1 and i == 0:
                    n_fp += 1

                n_total += 1
        if flag_Polyaxon == "True":
            logger.info('correct {},{}'.format(n_correct, n_correct / n_total))
            logger.info("true positive {}, {}".format(n_tp, n_tp / n_total))
            logger.info("false positive {}, {}".format(n_fp, n_fp / n_total))
            logger.info("false negative {} {}".format(n_fn, n_fn / n_total))
            logger.info("true negative {}, {}".format(n_tn, n_tn / n_total))
            logger.info('total {}'.format(n_total))
        else:
            print('correct', n_correct, n_correct / n_total)
            print("true positive", n_tp, n_tp / n_total)
            print("false positive", n_fp, n_fp / n_total)
            print("false negative", n_fn, n_fn / n_total)
            print("true negative", n_tn, n_tn / n_total)
            print('total', n_total)

            logger.info('correct {},{}'.format(n_correct, n_correct / n_total))
            logger.info("true positive {}, {}".format(n_tp, n_tp / n_total))
            logger.info("false positive {}, {}".format(n_fp, n_fp / n_total))
            logger.info("false negative {} {}".format(n_fn, n_fn / n_total))
            logger.info("true negative {}, {}".format(n_tn, n_tn / n_total))
            logger.info('total {}'.format(n_total))

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

        for i, test_db in enumerate([gap_list, non_gap_list, sacrum_list]):
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
                if round(prob_vertebra) == 1 and i == 1:
                    n_sp_sp += 1

                ###truenegative
                if round(prob_gap) == 1 and i == 0:
                    n_gap_gap += 1

                # sacrum
                if round(prob_sacrum) == 1 and i == 2:
                    n_sacrum_sacrum += 1

                # true spinous, false gap
                if round(prob_gap) == 1 and i == 1:
                    n_sp_gap += 1

                # true sacrum, false gap
                if round(prob_gap) == 1 and i == 2:
                    n_sacrum_gap += 1

                # true gap, false spinous
                if round(prob_vertebra) == 1 and i == 0:
                    n_gap_sp += 1

                # true sacrum, false spinous
                if round(prob_vertebra) == 1 and i == 2:
                    n_sacrum_sp += 1

                # true gap, false sacrum
                if round(prob_sacrum) == 1 and i == 0:
                    n_gap_sacrum += 1

                # true spinous, false sacrum
                if round(prob_sacrum) == 1 and i == 1:
                    n_sp_sacrum += 1

                n_total += 1

        n_correct = n_gap_gap + n_sp_sp + n_sacrum_sacrum

        if flag_Polyaxon == "True":
            logger.info('correct {} ({})'.format(n_correct, n_correct / n_total))
            logger.info("true gap predicted spinous {} ({})".format(n_gap_sp, n_gap_sp / n_total))
            logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
            logger.info("true spinous predicted gap {} ({})".format(n_sp_gap, n_sp_gap / n_total))
            logger.info("true spinous predicted sacrum {} ({})".format(n_sp_sacrum, n_sp_sacrum / n_total))
            logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
            logger.info("true sacrum predicted spinous {} ({})".format(n_sacrum_sp, n_sacrum_sp / n_total))
            logger.info('total {}'.format(n_total))
        else:
            print('correct', n_correct, n_correct / n_total)
            print("true gap predicted spinous", n_gap_sp, n_gap_sp / n_total)
            print("true gap predicted sacrum", n_gap_sacrum, n_gap_sacrum / n_total)
            print("true spinous predicted gap", n_sp_gap, n_sp_gap / n_total)
            print("true spinous predicted sacrum", n_sp_sacrum, n_sp_sacrum / n_total)
            print("true sacrum predicted gap", n_sacrum_gap, n_sacrum_gap / n_total)
            print("true sacrum predicted spinous", n_sacrum_sp, n_sacrum_sp / n_total)
            print('total', n_total)

            logger.info('correct {} ({})'.format(n_correct, n_correct / n_total))
            logger.info("true gap predicted spinous {} ({})".format(n_gap_sp, n_gap_sp / n_total))
            logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
            logger.info("true spinous predicted gap {} ({})".format(n_sp_gap, n_sp_gap / n_total))
            logger.info("true spinous predicted sacrum {} ({})".format(n_sp_sacrum, n_sp_sacrum / n_total))
            logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
            logger.info("true sacrum predicted spinous {} ({})".format(n_sacrum_sp, n_sacrum_sp / n_total))
            logger.info('total {}'.format(n_total))

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
                prob_thoracic = prob[0, 3]
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

        n_correct = n_gap_gap + n_lumbar_lumbar + n_sacrum_sacrum + n_thoracic_thoracic

        if flag_Polyaxon == "True":
            logger.info('correct {} ({})'.format(n_correct, n_correct / n_total))
            logger.info("true gap predicted lumbar {} ({})".format(n_gap_lumbar, n_gap_lumbar / n_total))
            logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
            logger.info("true gap predicted thoracic {} ({})".format(n_gap_thoracic, n_gap_thoracic / n_total))

            logger.info("true lumbar predicted gap {} ({})".format(n_lumbar_gap, n_lumbar_gap / n_total))
            logger.info("true lumbar predicted sacrum {} ({})".format(n_lumbar_sacrum, n_lumbar_sacrum / n_total))
            logger.info("true lumbar predicted thoracic {} ({})".format(n_lumbar_thoracic, n_lumbar_thoracic / n_total))

            logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
            logger.info("true sacrum predicted lumbar {} ({})".format(n_sacrum_lumbar, n_sacrum_lumbar / n_total))
            logger.info("true sacrum predicted thoracic {} ({})".format(n_sacrum_thoracic, n_sacrum_thoracic / n_total))

            logger.info("true thoracic predicted gap {} ({})".format(n_thoracic_gap, n_thoracic_gap / n_total))
            logger.info("true thoracic predicted lumbar {} ({})".format(n_thoracic_lumbar, n_thoracic_lumbar / n_total))
            logger.info("true thoracic predicted sacrum {} ({})".format(n_thoracic_sacrum, n_thoracic_sacrum / n_total))

            logger.info('total {}'.format(n_total))
        else:
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

            logger.info('correct {} ({})'.format(n_correct, n_correct / n_total))
            logger.info("true gap predicted lumbar {} ({})".format(n_gap_lumbar, n_gap_lumbar / n_total))
            logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
            logger.info("true gap predicted thoracic {} ({})".format(n_gap_thoracic, n_gap_thoracic / n_total))

            logger.info("true lumbar predicted gap {} ({})".format(n_lumbar_gap, n_lumbar_gap / n_total))
            logger.info("true lumbar predicted sacrum {} ({})".format(n_lumbar_sacrum, n_lumbar_sacrum / n_total))
            logger.info("true lumbar predicted thoracic {} ({})".format(n_lumbar_thoracic, n_lumbar_thoracic / n_total))

            logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
            logger.info("true sacrum predicted lumbar {} ({})".format(n_sacrum_lumbar, n_sacrum_lumbar / n_total))
            logger.info("true sacrum predicted thoracic {} ({})".format(n_sacrum_thoracic, n_sacrum_thoracic / n_total))

            logger.info("true thoracic predicted gap {} ({})".format(n_thoracic_gap, n_thoracic_gap / n_total))
            logger.info("true thoracic predicted lumbar {} ({})".format(n_thoracic_lumbar, n_thoracic_lumbar / n_total))
            logger.info("true thoracic predicted sacrum {} ({})".format(n_thoracic_sacrum, n_thoracic_sacrum / n_total))

            logger.info('total {}'.format(n_total))
def initialize_model(model_name, num_classes, feature_extract, use_pretrained):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# plt.ioff()
# plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')

    parser.add_argument('--flag_Polyaxon', type= str, default="False",
                        help='TRUE - Polyaxon, False - CPU')

    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--dataset', type=str, default='data1', metavar='N',
                        help='cross validation dataset')

    parser.add_argument('--network', default='resnet', type=str, metavar='N',
                        help='resnet, alexnet, vgg, squeezenet, densenet, inception')

    parser.add_argument('--use_pretrained', default='True', type=str,
                        help='True - Imagenet initialized weights, False - train from scratch')

    parser.add_argument('--Feature_extractation', default='False',type=str,
                        help='True - fine tune last layer only, False - update weights')

    parser.add_argument('--info_experiment', type=str, default="False",
                        help='description of the experiment')

    args = parser.parse_args()

    Polyaxon_flag = args.flag_Polyaxon




    if Polyaxon_flag == "True":
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        data_paths = get_data_paths()
        outputs_path = get_outputs_path()
        data_dir = os.path.join(data_paths['data1'], "SpinousProcessData",'PolyU_dataset')


        test_dir = os.path.join(data_dir, 'data_19subj', "test")

        dataset = args.dataset
        data_dir = os.path.join(data_dir, dataset)




        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(outputs_path,'SpinousProcess.log')))

        # Polyaxon
        experiment = Experiment()

    else:

        data_dir = "CNN_regions_data_24subj"
        test_dir = "CNN_regions_data_24subj/val"



        now = datetime.now()
        print("now =", now)
        # logger.info("now = {}".format(now))

        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d_%m_%Y %H_%M_%S")
        # print("date and time =", dt_string)

        OUTPUT_PATH = os.path.join("runs", '%s' % dt_string)
        outputs_path = OUTPUT_PATH
        print("output path", OUTPUT_PATH)
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)
        os.chmod(OUTPUT_PATH, stat.S_IRWXO)
        os.chmod(os.path.join("runs"), stat.S_IRWXO)

        config.BATCH_SIZE = args.batch_size

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(OUTPUT_PATH, 'SpinousProcess.log')))

        logger.info("output dir {}".format(OUTPUT_PATH))

    # else:
    #     data_dir = '/media/maryviktory/My Passport/IPCAI 2020 TUM/DATA_toNas_for CNN_IPCAI/data_all(15patients train, 4 test)/'
    #     data_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\CNN_regions_data_24subj"
    #     outputs_path = 'D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\CNN_regions_data_24subj'
    #     logger = logging.getLogger()
    #     logger.setLevel(logging.INFO)
    #     logger.addHandler(logging.FileHandler(os.path.join(outputs_path, 'SpinousProcess.log')))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)







    # Make iterable objects with the dataloaders


    # need to preserve some information about our dataset,
    # specifically the size of the dataset and the names of the classes in our dataset.



    #device - a CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Grab some of the training data to visualize
    # inputs, classes, path = next(iter())
    # if Polyaxon_flag == "True":
    #     logger.info(outputs_path)
    # else:
    #     print(outputs_path)

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)
##########  Parse data
    model_name = args.network
    use_pretrained = args.use_pretrained
    if use_pretrained=="True":
        use_pretrained = True
    else:
        use_pretrained = False

    feature_extract = args.Feature_extractation
    if feature_extract=="True":
        feature_extract = True
    else:
        feature_extract = False

    dataset = args.dataset

    num_epochs = args.epochs


#######3 Chhose model
    trainloader, valloader, num_classes = CNN_load_train_val(data_dir,config)
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained)

    # print(model_ft)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    logger.info("learning rate: {}".format(args.lr))
    logger.info("Pretrained network: {}".format(use_pretrained))
    logger.info("Batch size: {}".format(config.BATCH_SIZE))
    logger.info("classes: {}".format(os.listdir(os.path.join(data_dir,"train"))))
    print("Batch size: {}".format(config.BATCH_SIZE))


    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

    # Train and evaluate
    model_ft,model_save_path= train_model(trainloader,valloader,model_ft, criterion, optimizer_ft, exp_lr_scheduler, outputs_path, num_epochs, model_name,feature_extract, use_pretrained,dataset)

    print("model path",model_save_path)
    test_model(test_dir, model_save_path,num_classes, Polyaxon_flag)

############### FINE TUNING #################

    # model_ft = models.densenet121(pretrained=True)
    # # print(model_ft)
    #
    #
    # #fc - fully connected layer, the last one
    # num_ftrs = model_ft.classifier.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft.classifier = nn.Linear(num_ftrs, 2)
    #
    #
    # for item in model_ft.parameters():
    #     item.requires_grad = False
    #     # logger.info(item + 'has been unfrozen.')
    #
    # model_ft = model_ft.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.05, momentum=0.9)
    #
    # # Decay LR by a factor of 0.1 every 3 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    #
    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, outputs_path,
    #                        num_epochs=50)

    #
    # # visualize_model(model_ft)


############## FIXED FEATURE EXTRACTOR ###############
    # # Note that the parameters of imported models are set to requires_grad=True by default
    # model_conv = models.densenet121(pretrained=True)
    # for name, child in model_conv.named_children():
    #     if name in ['layer2','layer3', 'layer4','avgpool']:
    #         logger.info(name + 'has been unfrozen.')
    #         for param in child.parameters():
    #             param.requires_grad = True
    #     else:
    #         for param in child.parameters():
    #             param.requires_grad = False
    #
    # # Parameters of newly constructed modules have requires_grad=True by default
    # num_ftrs = model_conv.fc.in_features
    # model_conv.fc = nn.Linear(num_ftrs, 2)
    #
    # model_conv = model_conv.to(device)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # # Observe that only parameters of final layer are being optimized as
    # # opposed to before.
    # optimizer_conv = torch.optim.SGD(filter(lambda x: x.requires_grad, model_conv.parameters()), lr=0.0005, momentum=0.9)
    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #
    # model_conv = train_model(model_conv, criterion, optimizer_conv,
    #                          exp_lr_scheduler,outputs_path, num_epochs=100)

