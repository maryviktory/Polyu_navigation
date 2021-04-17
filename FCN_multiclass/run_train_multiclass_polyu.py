import FCN_multiclass.sp_utils as utils
# from sp_utils.config import config
from FCN_multiclass.sp_utils.config import config
import argparse
import logging
import os
# from torchsummary import summary
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchvision as tv
from torch.optim import lr_scheduler

def run_val(model, valloader, device, criterion, writer, epoch, config):
    val_loss = 0
    acc = utils.AverageMeter()
    model.eval()
    running_loss=0
    running_corrects=0
    num = 0
    with torch.no_grad():
        for i, (inputs, labels,class_id) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
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
            batch_loss = loss_classification + loss_heatmap

            running_loss += batch_loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred_class == class_id.data)

            # batch_loss = criterion(logps, labels.float())
            val_loss += batch_loss.item()

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

        writer.add_scalar('Loss/val', float(val_loss / len(valloader)), epoch)
        # writer.add_scalar('Accuracy/val', float(accuracy / len(valloader)), epoch)
        writer.add_scalar('Accuracy/val', acc.avg, epoch)

        epoch_loss = running_loss / num
        epoch_acc_classification = running_corrects.double() / num
        print("Classification Validation Accuracy {} epoch: {}, loss: {} ").format(epoch, epoch_acc_classification,
                                                                              epoch_loss)

    return acc.avg

def main(config):
    logging.basicConfig(level=logging.INFO)
    logging.info("STARTING PROGRAM")

    if config.TRAIN.POLYAXON:
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path
        data_dir = get_data_paths()
        config.DATASET.OUTPUT_PATH = get_outputs_path()
        config.DATASET.PATH  = os.path.join(data_dir['data1'], config.DATASET.PATH_NAS)
        model_path = os.path.join(data_dir['data1'],config.MODEL.PRETRAINED_NAS)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(config.DATASET.OUTPUT_PATH, 'Heatmaps_from_human_joints.log')))

        # Polyaxon
        experiment = Experiment()

    else:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(config.DATASET.OUTPUT_PATH , 'Heatmaps_Resnet101.log')))
        model_path = config.MODEL.PRETRAINED

    trainloader, valloader = utils.multiclass_heatmap_load_train_val(config.DATASET.PATH, "train", "validation", config)
    # print(trainloader)
    logger.info('batch size {}'.format(config.TRAIN.BATCH_SIZE))
    print('dataset NAS',config.DATASET.PATH_NAS)
    logger.info("weights {}".format(config.TRAIN.UPDATE_WEIGHTS))
    logger.info("Model: {}".format(model_path))
    logger.info("LR: {}".format(config.TRAIN.LR))
    model = utils.model_pose_resnet.get_pose_net(model_path,is_train = True)

    model.eval()

    for name,parameter in model.named_parameters():
        parameter.requires_grad = config.TRAIN.UPDATE_WEIGHTS
        if "deconv" in name or "final" in name:
            parameter.requires_grad = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    model.to(device)

    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)

    writer = SummaryWriter(config.DATASET.OUTPUT_PATH)
    best_acc = 0

    for epoch in range(config.TRAIN.END_EPOCH):
        criterion = nn.MSELoss()
        logger.info('Epoch {}/{}'.format(epoch, config.TRAIN.END_EPOCH - 1))
        logger.info('-' * 10)
        acc = utils.AverageMeter()
        acc_classification = utils.AverageMeter()
        batch_loss = utils.AverageMeter()
        running_loss = 0.0
        running_corrects = 0
        num = 0
        for i, (inputs, labels,class_id) in enumerate(trainloader):

            # print(class_id)
            inputs, labels = inputs.to(device), labels.to(device)
            num_images = inputs.size()[0]
            # print(summary(model, tuple(inputs.size())[1:]))
            logps,multiclass = model.forward(inputs)
            # print("multiclass: ", multiclass)

            probabilities = torch.nn.functional.softmax(multiclass, dim=0)
            # print("probabilities", probabilities)
            _, pred_class = torch.max(multiclass, 1)
            # print("preds_class", pred_class)

            # print("labels",labels)
            criterion_classification = nn.CrossEntropyLoss()
            loss_classification = criterion_classification(multiclass,class_id)

            criterion_heatmap = nn.MSELoss()
            loss_heatmap = criterion_heatmap(logps, labels.float())

            loss = loss_classification+loss_heatmap
            batch_loss.update(loss.item(),inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(pred_class == class_id.data)
            batch_correct = torch.sum(pred_class == class_id.data)
            batch_acc_class = batch_correct/config.TRAIN.BATCH_SIZE
            # print("running corrects",running_corrects.size())

            _, avg_acc, cnt, pred,target,dists = utils.accuracy(logps.detach().cpu().numpy(),
                                             labels.detach().cpu().numpy(),thr = config.TRAIN.THRESHOLD)
            print("Current batch accuracy heatmap: ", avg_acc)
            acc.update(avg_acc,cnt)
            # acc_classification.update()
            print("Batch {} train accurcy: {}, classification acc: {}, loss: {}".format(i, acc.avg, batch_acc_class,batch_loss.avg))
            num = num + num_images

        writer.add_scalar('Loss/train', float(batch_loss.avg), epoch)

        epoch_loss = running_loss / (num)
        epoch_acc_classification = running_corrects.double() / (num)
        logger.info("Classification Accuracy {} epoch: {}, loss: {} ").format(epoch,epoch_acc_classification,epoch_loss)

        val_acc = run_val(model, valloader, device, criterion, writer, epoch,config)

        logger.info('Train Loss: {:.4f} Train Acc: {:.4f} Val Acc: {:.4f}'.format(
             batch_loss.avg, acc.avg, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            logging.info("best val at epoch: " + str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "best_model.pt"))

        if epoch % 250 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "model" + str(epoch) + ".pt"))

    logger.info('Best val Acc: {:4f}'.format(best_acc))

def Parser():
    parser = argparse.ArgumentParser(description='DeepSpine script')

    parser.add_argument('--data_dir', type=str, default="SpinousProcessData/FCN_PWH_train_dataset_heatmaps/data_19subj_2", metavar='N',
                        help='')

    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--update_weights', type=bool, default=False, metavar='P',
                        help='whether to train the networs from scratches or with fine tuning')

    parser.add_argument('--lr', type=float, default=0.001, metavar='BS',
                        help='learning rate')
    args = parser.parse_args()

    return args

def update_config(config,args):
    if args.data_dir:
        config.DATASET.PATH_NAS = args.data_dir
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.update_weights:
        config.TRAIN.UPDATE_WEIGHTS = args.update_weights
    if args.lr:
        config.TRAIN.LR = args.lr

if __name__ == '__main__':
    args = Parser()
    update_config(config,args)
    main(config)