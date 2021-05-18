import sp_utils as utils
# from sp_utils.config import config
from sp_utils.config import config
import argparse
import logging
import os
import stat
from datetime import datetime
# from torchsummary import summary
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchvision as tv
from torch.optim import lr_scheduler
import time
# import matplotlib.pyplot as plt

def run_val(model, valloader, device, criterion, writer, epoch, config,logger):
    val_loss = 0
    acc = utils.AverageMeter()
    acc_second_head = utils.AverageMeter()
    batch_loss = utils.AverageMeter()
    class_loss = utils.AverageMeter()
    heatmap_loss = utils.AverageMeter()
    heatmap_second_head_loss = utils.AverageMeter()
    model.eval()
    running_loss=0
    running_corrects=0
    num = 0
    with torch.no_grad():
        if config.TRAIN.three_heads == True:
            for i, (inputs, labels,labels_sacrum, class_id) in enumerate(valloader):
                inputs, labels,labels_sacrum, class_id = inputs.to(device), labels.to(device), labels_sacrum.to(device), class_id.to(device)
                num_images = inputs.size()[0]
                logps,logps_second_head, multiclass = model.forward(inputs)
                # multiclass = torch.sigmoid(multiclass).numpy()
                # probabilities = torch.nn.functional.softmax(multiclass, dim=0)
                # print("probabilities",probabilities)
                _, pred_class = torch.max(multiclass, 1)
                # print("preds_class", pred_class)

                criterion_classification = nn.CrossEntropyLoss()
                loss_classification = criterion_classification(multiclass, class_id)
                criterion_heatmap = nn.MSELoss()
                loss_heatmap = criterion_heatmap(logps, labels.float())
                loss_heatmap_second_head = criterion_heatmap(logps_second_head, labels_sacrum.float())

                loss = loss_classification + config.TRAIN.loss_alpha * loss_heatmap+ config.TRAIN.loss_alpha * loss_heatmap_second_head

                batch_loss.update(loss.item(), inputs.size(0))
                class_loss.update(loss_classification.item(), inputs.size(0))
                heatmap_loss.update(loss_heatmap.item(), inputs.size(0))
                heatmap_second_head_loss.update(loss_heatmap_second_head.item(), inputs.size(0))

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_class == class_id.data)

                # batch_loss = criterion(logps, labels.float())
                # val_loss += batch_loss.item()

                _, avg_acc, cnt, pred, target, dists = utils.accuracy(logps.detach().cpu().numpy(),
                                                                      labels.detach().cpu().numpy(),
                                                                      thr=config.TRAIN.THRESHOLD)

                acc.update(avg_acc, cnt)

                _, avg_acc_second_head, cnt_2, _, _, _ = utils.accuracy(logps_second_head.detach().cpu().numpy(),
                                                                        labels_sacrum.detach().cpu().numpy(),
                                                                        thr=config.TRAIN.THRESHOLD)
                acc_second_head.update(avg_acc_second_head, cnt_2)

                ps = torch.sigmoid(logps).float()
                s_out = logps.float()

                if epoch % 10 and i == 2:
                    grid_images = tv.utils.make_grid(inputs)
                    writer.add_image('images', grid_images, epoch)

                    grid_labels = tv.utils.make_grid(labels)
                    writer.add_image('labels', grid_labels, epoch)

                    grid_output = tv.utils.make_grid(ps)
                    writer.add_image('output', grid_output, epoch)

                    grid_output_sig = tv.utils.make_grid(s_out)
                    writer.add_image('output', grid_output_sig, epoch)

                num = num + num_images
        else:
            for i, (inputs, labels,class_id) in enumerate(valloader):
                inputs, labels, class_id = inputs.to(device), labels.to(device), class_id.to(device)
                num_images = inputs.size()[0]
                logps,multiclass = model.forward(inputs)
                # multiclass = torch.sigmoid(multiclass).numpy()
                # probabilities = torch.nn.functional.softmax(multiclass, dim=0)
                # print("probabilities",probabilities)
                _, pred_class = torch.max(multiclass, 1)
                # print("preds_class", pred_class)

                criterion_classification = nn.CrossEntropyLoss()
                loss_classification = criterion_classification(multiclass, class_id)
                criterion_heatmap = nn.MSELoss()
                loss_heatmap = criterion_heatmap(logps, labels.float())
                loss = loss_classification + config.TRAIN.loss_alpha*loss_heatmap

                batch_loss.update(loss.item(), inputs.size(0))
                class_loss.update(loss_classification.item(), inputs.size(0))
                heatmap_loss.update(loss_heatmap.item(), inputs.size(0))

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_class == class_id.data)

                # batch_loss = criterion(logps, labels.float())
                # val_loss += batch_loss.item()

                _, avg_acc, cnt, pred,target,dists = utils.accuracy(logps.detach().cpu().numpy(),
                                                 labels.detach().cpu().numpy(),thr = config.TRAIN.THRESHOLD)

                acc.update(avg_acc, cnt)

                ps = torch.sigmoid(logps).float()
                s_out = logps.float()


                if epoch % 10 and i == 2:

                    grid_images = tv.utils.make_grid(inputs)
                    writer.add_image('images', grid_images, epoch)

                    grid_labels = tv.utils.make_grid(labels)
                    writer.add_image('labels', grid_labels, epoch)

                    grid_output = tv.utils.make_grid(ps)
                    writer.add_image('output', grid_output, epoch)

                    grid_output_sig = tv.utils.make_grid(s_out)
                    writer.add_image('output', grid_output_sig, epoch)

                num = num + num_images

        epoch_loss = running_loss / num
        epoch_acc_classification = running_corrects.double() / num

        mean_accuracy = (acc.avg + epoch_acc_classification) / 2
        writer.add_scalar('Loss/total_val', float(batch_loss.avg), epoch)
        writer.add_scalar('Loss_class/val', float(class_loss.avg), epoch)
        writer.add_scalar('Loss_heatmap/val', float(heatmap_loss.avg), epoch)

        # writer.add_scalar('Accuracy/val', float(accuracy / len(valloader)), epoch)
        writer.add_scalar('Accuracy/heatmap_val', acc.avg, epoch)
        writer.add_scalar('Accuracy/class_val', epoch_acc_classification, epoch)
        logger.info("Total validation loss {}, total accuracy {}".format(batch_loss.avg, mean_accuracy))

        logger.info("Classification Validation, epoch: {}, Accuracy {} , loss: {} ".format(epoch, epoch_acc_classification,class_loss.avg))
        logger.info("Heatmap Validation (epoch {}),Accuracy {} , loss: {} ".format(epoch,acc.avg,heatmap_loss.avg))

        if config.TRAIN.three_heads == True:
            writer.add_scalar('Loss_heatmap_sacrum/val', float(heatmap_second_head_loss.avg), epoch)
            writer.add_scalar('Accuracy/heatmap_sacrum_val', float(acc_second_head.avg), epoch)
            logger.info(
                "Heatmap sacrum Validation (epoch {}),Accuracy {} , loss: {} ".format(epoch, acc_second_head.avg, heatmap_second_head_loss.avg))
            mean_accuracy = (acc.avg + epoch_acc_classification+acc_second_head.avg) / 3

    return acc.avg,mean_accuracy, epoch_acc_classification

def main(config):
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO)
    logging.info("STARTING PROGRAM")


    # datetime object containing current date and time
    now = datetime.now()

    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string= now.strftime("%d_%m_%Y %H_%M_%S")
    # print("date and time =", dt_string)


    config.DATASET.OUTPUT_PATH = os.path.join(config.DATASET.PATH ,"runs",'%s'%dt_string)
    if not os.path.exists(config.DATASET.OUTPUT_PATH):
        os.makedirs(config.DATASET.OUTPUT_PATH)
    os.chmod(config.DATASET.OUTPUT_PATH,stat.S_IRWXO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(config.DATASET.OUTPUT_PATH,'Heatmaps_Resnet101.log')))
    model_path = os.path.join(config.DATASET.PATH,"pretrained model","model_best_resnet_fixed_False_pretrained_True_Multiclass_spine_4classes_exp6106.pt")

    trainloader, valloader,class_num = utils.multiclass_heatmap_load_train_val(config.DATASET.PATH, "train", "validation", config)

    logger.info("config.TRAIN.three_heads = {}".format(config.TRAIN.three_heads))
    logger.info("class_num from DataLoader {}".format(class_num))
    logger.info('batch size {}'.format(config.TRAIN.BATCH_SIZE))
    print('dataset',config.DATASET.PATH)
    logger.info("weights {}".format(config.TRAIN.UPDATE_WEIGHTS))
    logger.info("weights loss {}".format(config.TRAIN.loss_alpha))
    logger.info("Model: {}".format(model_path))
    logger.info("LR: {}".format(config.TRAIN.LR))
    logger.info("Imagenet: {}".format(config.MODEL.Imagenet_pretrained))
    logger.info("Adaptive weights: {}".format(config.TRAIN.adaptive_weights))
    logger.info("Weight_decay: {}".format(config.TRAIN.weight_decay))
    logger.info("epoch: {}".format(config.TRAIN.END_EPOCH))

    model = utils.model_pose_resnet.get_pose_net(config,model_path,is_train = True)

    model.eval()
    # print(model)
    for name,parameter in model.named_parameters():
        parameter.requires_grad = config.TRAIN.UPDATE_WEIGHTS
        if "deconv" in name or "final" in name or "fc_class" in name:
            parameter.requires_grad = True

    for name,parameter in model.named_parameters():
        if parameter.requires_grad == True:
            print(name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config.TRAIN.adaptive_weights = False
    if config.TRAIN.adaptive_weights == True:
        # log_vars = nn.Parameter(torch.zeros((2)))
        log_var_a = torch.zeros((1,), requires_grad=True,device=device)
        log_var_b = torch.zeros((1,), requires_grad=True,device=device)
        # print("model parameters",list(model.parameters()))
        # get all parameters (model parameters + task dependent log variances)
        params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b])
        optimizer = optim.Adam(params)

        # optimizer = optim.Adam([
        #     {"params": model.conv1.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.bn1.parameters(),"lr": config.TRAIN.LR},
        #     {"params": model.relu.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.maxpool.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.layer1.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.layer2.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.layer3.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.layer4.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.avgpool_class.parameters(), "lr": config.TRAIN.LR_cl},
        #     {"params": model.fc_class.parameters(), "lr": config.TRAIN.LR_cl},
        #     {"params": model.deconv_layers.parameters(), "lr": config.TRAIN.LR},
        #     {"params": model.final_layer.parameters(), "lr": config.TRAIN.LR},
        #     {"params": log_vars, "lr": config.TRAIN.LR},
        # ], lr=config.TRAIN.LR)
    else:

        if config.TRAIN.weight_decay >0.0:
            optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR,weight_decay=config.TRAIN.weight_decay)  # weight_decay=config.TRAIN.weight_decay
            logger.info("Using weight decay")
        else:

            optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR)


            # logger.info("Using discriminative learning rates")
            # optimizer = optim.Adam([
            #         {"params": model.conv1.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.bn1.parameters(),"lr": config.TRAIN.LR},
            #         {"params": model.relu.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.maxpool.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.layer1.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.layer2.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.layer3.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.layer4.parameters(), "lr": config.TRAIN.LR},
            #         {"params": model.avgpool_class.parameters(), "lr": config.TRAIN.LR_cl},
            #         {"params": model.fc_class.parameters(), "lr": config.TRAIN.LR_cl},
            #         {"params": model.deconv_layers.parameters(), "lr": config.TRAIN.LR_heatmap},
            #         {"params": model.final_layer.parameters(), "lr": config.TRAIN.LR_heatmap},
            #
            #     ], lr=config.TRAIN.LR)
    model.to(device)
    print("Model on cuda: ", next(model.parameters()).is_cuda)
    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    writer = SummaryWriter(config.DATASET.OUTPUT_PATH)
    best_acc = 0
    best_acc_class = 0
    best_mean_acc = 0
    initial_time = time.time()
    for epoch in range(config.TRAIN.END_EPOCH):
        # for g in optimizer.param_groups:
        #     print(g['lr'])

        logger.info('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        logger.info('Epoch-{} lr body {} lr_cl {}, lr_heat {}'.format(epoch,  config.TRAIN.LR,config.TRAIN.LR_cl, config.TRAIN.LR_heatmap))

        criterion = nn.MSELoss()
        logger.info('Epoch {}/{}'.format(epoch, config.TRAIN.END_EPOCH - 1))
        logger.info('-' * 10)
        acc = utils.AverageMeter()
        acc_second_head = utils.AverageMeter()
        acc_classification = utils.AverageMeter()
        batch_loss = utils.AverageMeter()
        class_loss = utils.AverageMeter()
        heatmap_loss = utils.AverageMeter()
        heatmap_second_head_loss = utils.AverageMeter()
        loss_weight = utils.AverageMeter()
        running_loss = 0.0
        running_corrects = 0
        num = 0
        class_2 = 0
        class_1 = 0
        class_0 = 0
        class_3 = 0

        if epoch == 25:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = True

            for g in optimizer.param_groups:
                g['lr'] = 0.0001
            logger.info("UNFIX PARAMETERS GRAD")

        if config.TRAIN.three_heads == True:
            for i, (inputs, labels,labels_sacrum, class_id) in enumerate(trainloader):
                for id in class_id:
                    if id == 2:
                        class_2 = class_2 + 1
                    if id == 1:
                        class_1 = class_1 + 1
                    if id == 0:
                        class_0 = class_0 + 1
                    if id == 3:
                        class_3 = class_3 + 1

                # grid = tv.utils.make_grid(inputs)
                # plt.imshow(grid.numpy().transpose((1, 2, 0)))

                # imshow(inputs.permute(1, 2, 0))
                # print(class_id)
                # plt.show()
                inputs, labels, labels_sacrum,class_id = inputs.to(device), labels.to(device),labels_sacrum.to(device), class_id.to(device)
                num_images = inputs.size()[0]
                # print(summary(model, tuple(inputs.size())[1:]))
                logps, logps_second_head,multiclass = model.forward(inputs)

                # print("multiclass: ", multiclass)

                probabilities = torch.nn.functional.softmax(multiclass, dim=0)
                # print("probabilities", probabilities)
                _, pred_class = torch.max(multiclass, 1)
                # print("preds_class", pred_class)

                # print("labels",labels)

                if config.TRAIN.adaptive_weights == True:
                    loss = utils.MultiTaskLossWrapper(log_var_a, log_var_b)
                    loss_classification, loss_heatmap = loss.forward(multiclass, logps, class_id, labels)
                    loss = loss_classification + loss_heatmap
                    # loss = loss/2
                else:
                    criterion_classification = nn.CrossEntropyLoss()
                    loss_classification = criterion_classification(multiclass, class_id)

                    criterion_heatmap = nn.MSELoss()
                    loss_heatmap = criterion_heatmap(logps, labels.float())
                    loss_heatmap_second_head = criterion_heatmap(logps_second_head, labels_sacrum.float())

                    # loss_weight.update(loss_alpha)
                    # print("weight loss",config.TRAIN.loss_alpha)
                    # config.TRAIN.loss_alpha = loss_alpha
                    loss = loss_classification + config.TRAIN.loss_alpha * loss_heatmap + config.TRAIN.loss_alpha * loss_heatmap_second_head

                class_loss.update(loss_classification.item(), inputs.size(0))
                heatmap_loss.update(loss_heatmap.item(), inputs.size(0))
                heatmap_second_head_loss.update(loss_heatmap_second_head.item(),inputs.size(0))
                # loss = config.TRAIN.loss_alpha*loss_heatmap
                batch_loss.update(loss.item(), inputs.size(0))

                optimizer.zero_grad()
                loss.backward()  # retain_graph=True
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_class == class_id.data)

                # print("running corrects",running_corrects.size())

                _, avg_acc, cnt, pred, target, dists = utils.accuracy(logps.detach().cpu().numpy(),
                                                                      labels.detach().cpu().numpy(),
                                                                      thr=config.TRAIN.THRESHOLD)
                # print("Current batch accuracy heatmap: ", avg_acc)
                acc.update(avg_acc, cnt)

                _, avg_acc_second_head, cnt_2, _, _, _ = utils.accuracy(logps_second_head.detach().cpu().numpy(),
                                                                      labels_sacrum.detach().cpu().numpy(),
                                                                      thr=config.TRAIN.THRESHOLD)
                # print("Current batch accuracy heatmap: ", avg_acc)
                acc_second_head.update(avg_acc_second_head, cnt_2)


                # acc_classification.update()
                # print("Batch {} train accurcy: {}, classification acc: {}, loss: {}".format(i, acc.avg, batch_acc_class,batch_loss.avg))
                num = num + num_images

        #NOTE: For 2 heads model
        else:
            for i, (inputs, labels,class_id) in enumerate(trainloader):
                # print(inputs.shape)

                for id in class_id:
                    if id == 2:
                        class_2 = class_2+1
                    if id ==1:
                        class_1 = class_1 +1
                    if id == 0:
                        class_0 = class_0+1
                    if id ==3:
                        class_3 = class_3 +1

                # grid = tv.utils.make_grid(inputs)
                # plt.imshow(grid.numpy().transpose((1, 2, 0)))

                # imshow(inputs.permute(1, 2, 0))
                # print(class_id)
                # plt.show()
                inputs, labels, class_id = inputs.to(device), labels.to(device), class_id.to(device)
                num_images = inputs.size()[0]
                # print(summary(model, tuple(inputs.size())[1:]))
                logps,multiclass = model.forward(inputs)

                # print("multiclass: ", multiclass)

                probabilities = torch.nn.functional.softmax(multiclass, dim=0)
                # print("probabilities", probabilities)
                _, pred_class = torch.max(multiclass, 1)
                # print("preds_class", pred_class)

                # print("labels",labels)


                if config.TRAIN.adaptive_weights == True:
                    loss = utils.MultiTaskLossWrapper(log_var_a, log_var_b)
                    loss_classification, loss_heatmap = loss.forward(multiclass,logps,class_id,labels)
                    loss = loss_classification + loss_heatmap
                    # loss = loss/2
                else:
                    criterion_classification = nn.CrossEntropyLoss()
                    loss_classification = criterion_classification(multiclass,class_id)

                    criterion_heatmap = nn.MSELoss()
                    loss_heatmap = criterion_heatmap(logps, labels.float())


                    # loss_weight.update(loss_alpha)
                    # print("weight loss",config.TRAIN.loss_alpha)
                    # config.TRAIN.loss_alpha = loss_alpha
                    loss = loss_classification+config.TRAIN.loss_alpha*loss_heatmap

                class_loss.update(loss_classification.item(), inputs.size(0))
                heatmap_loss.update(loss_heatmap.item(), inputs.size(0))
                # loss = config.TRAIN.loss_alpha*loss_heatmap
                batch_loss.update(loss.item(),inputs.size(0))


                optimizer.zero_grad()
                loss.backward() #retain_graph=True
                optimizer.step()




                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_class == class_id.data)

                # print("running corrects",running_corrects.size())

                _, avg_acc, cnt, pred,target,dists = utils.accuracy(logps.detach().cpu().numpy(),
                                                 labels.detach().cpu().numpy(),thr = config.TRAIN.THRESHOLD)
                # print("Current batch accuracy heatmap: ", avg_acc)
                acc.update(avg_acc,cnt)
                # acc_classification.update()
                # print("Batch {} train accurcy: {}, classification acc: {}, loss: {}".format(i, acc.avg, batch_acc_class,batch_loss.avg))
                num = num + num_images
        if config.TRAIN.adaptive_weights== True:
            print("log vars",log_var_a, log_var_b)
            std_1 = torch.exp(log_var_a) ** 0.5
            std_2 = torch.exp(log_var_b) ** 0.5
            print([std_1.item(), std_2.item()])

        # loss_alpha = class_loss.avg / heatmap_loss.avg
        # config.TRAIN.loss_alpha = loss_alpha
        # print("weight loss", loss_alpha)
        print("cl0",class_0)
        print("cl1",class_1)
        print("cl2", class_2)
        print("cl3", class_3)
        print("sceduler LR",exp_lr_scheduler.get_last_lr())
        epoch_loss = running_loss / (num)
        epoch_acc_classification = running_corrects.double() / (num)
        epoch_time = time.time() - initial_time

        writer.add_scalar('Accuracy/heatmap_train', float(acc.avg), epoch)
        writer.add_scalar('Accuracy/class_train', epoch_acc_classification, epoch)
        writer.add_scalar('Loss/total_train', float(batch_loss.avg), epoch)
        writer.add_scalar('Loss_class/train', float(class_loss.avg), epoch)
        writer.add_scalar('Loss_heatmap/train', float(heatmap_loss.avg), epoch)

        if config.TRAIN.three_heads ==True:
            writer.add_scalar('Loss_heatmap_sacrum/train', float(heatmap_second_head_loss.avg), epoch)
            writer.add_scalar('Accuracy/heatmap_sacrum_train', float(acc_second_head.avg), epoch)

        logger.info("epoch time: {}".format(epoch_time))
        logger.info('Classification Train Accuracy {} epoch: {}'.format(epoch,epoch_acc_classification))



        val_acc, mean_acc, val_acc_class = run_val(model, valloader, device, criterion, writer, epoch,config,logger)

        exp_lr_scheduler.step()
        logger.info("loss classification {}, loss heatmap {} ".format(class_loss.avg,heatmap_loss.avg))

        logger.info('Train Total Loss: {:.4f} Train Heatmap Acc: {:.4f} Val Heatmap Acc: {:.4f}'.format(
             batch_loss.avg, acc.avg, val_acc))
        if val_acc_class > best_acc_class:
            best_acc_class = val_acc_class
            logging.info("best val class at epoch: " + str(epoch) + str(best_acc_class))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "best_acc_class_model.pt"))


        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            logging.info("best val mean at epoch: " + str(epoch) + str(best_mean_acc))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "best_mean_acc_model.pt"))

        if val_acc > best_acc:
            best_acc = val_acc
            logging.info("best val heatmap at epoch: " + str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "best_model.pt"))

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "model" + str(epoch) + ".pt"))

    logger.info('Best val Acc: {:4f}'.format(best_acc))
    logger.info("Run time: {} ".format(time.time()-initial_time))

def Parser():
    parser = argparse.ArgumentParser(description='DeepSpine script')

    parser.add_argument('--data_dir', type=str, default="data_24subj_19train_multiclass_3head/", metavar='N',
                        help='')

    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epoch', type=int, default=50, metavar='N',
                        help='num of epochs to train')

    parser.add_argument('--weight_loss', type=int, default=500, metavar='N',
                        help='num of epochs to train')

    parser.add_argument('--update_weights', type=int, default=1,
                        help='1 - yes, 0- no, whether to train the networs from scratches or with fine tuning')

    parser.add_argument('--imagenet',  type=int, default=1,
                        help='1 - yes, 0 -no whether to train the networs from imagenet pretrained model or with 4 class pretrained')

    parser.add_argument('--lr', type=float, default=0.001, metavar='BS',
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, default=0.0, metavar='BS',
                        help='weight decay')

    args = parser.parse_args()

    return args

def update_config(config,args):
    if args.data_dir:
        config.DATASET.PATH = args.data_dir
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.update_weights==1:
        config.TRAIN.UPDATE_WEIGHTS = True
    if args.update_weights == 0:
        config.TRAIN.UPDATE_WEIGHTS = False
    if args.lr:
        config.TRAIN.LR = args.lr
    if args.imagenet==1:
        config.MODEL.Imagenet_pretrained = True
    if args.imagenet==0:
        config.MODEL.Imagenet_pretrained = False
    if args.epoch:
        config.TRAIN.END_EPOCH = args.epoch
    if args.weight_decay:
        config.TRAIN.weight_decay = args.weight_decay
    if args.weight_loss:
        config.TRAIN.loss_alpha = args.weight_loss

if __name__ == '__main__':
    args = Parser()
    update_config(config,args)
    main(config)

