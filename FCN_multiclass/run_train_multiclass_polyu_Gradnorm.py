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
import time

def run_val(model, valloader, device, criterion, writer, epoch, config,logger):
    val_loss = 0
    acc = utils.AverageMeter()
    batch_loss = utils.AverageMeter()
    class_loss = utils.AverageMeter()
    heatmap_loss = utils.AverageMeter()
    model.eval()
    running_loss=0
    running_corrects=0
    num = 0
    with torch.no_grad():
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


    return acc.avg,mean_accuracy, epoch_acc_classification

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

    trainloader, valloader,class_num = utils.multiclass_heatmap_load_train_val(config.DATASET.PATH, "train", "validation", config)
    logger.info("class_num from DataLoader {}".format(class_num))
    logger.info('batch size {}'.format(config.TRAIN.BATCH_SIZE))
    print('dataset NAS',config.DATASET.PATH_NAS)
    logger.info("weights {}".format(config.TRAIN.UPDATE_WEIGHTS))
    logger.info("weights loss {}".format(config.TRAIN.loss_alpha))
    logger.info("Model: {}".format(model_path))
    logger.info("LR: {}".format(config.TRAIN.LR))
    model = utils.model_pose_resnet.get_pose_net(model_path,is_train = True)


    model.eval()

    for name,parameter in model.named_parameters():
        parameter.requires_grad = config.TRAIN.UPDATE_WEIGHTS
        if "deconv" in name or "final" in name:
            parameter.requires_grad = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR,weight_decay=config.TRAIN.weight_decay) #weight_decay=config.TRAIN.weight_decay
    model.to(device)
    print("Model on cuda: ", next(model.parameters()).is_cuda)
    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1./2)

    writer = SummaryWriter(config.DATASET.OUTPUT_PATH)
    best_acc = 0
    best_acc_class = 0
    best_mean_acc = 0
    initial_time = time.time()
    criterion_classification = nn.CrossEntropyLoss()
    criterion_heatmap = nn.MSELoss()
    GradNorm = True
    if GradNorm == True:

        Weightloss1 = torch.FloatTensor([1]).clone().detach().requires_grad_(True)
        Weightloss2 = torch.FloatTensor([1]).clone().detach().requires_grad_(True)
        params = [Weightloss1, Weightloss2]
        # print("params", params)
        opt2 = torch.optim.Adam([Weightloss1,Weightloss2], lr=config.TRAIN.LR)
        Gradloss = nn.L1Loss()
        alph = 1


    for epoch in range(config.TRAIN.END_EPOCH):

        logger.info('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        criterion = nn.MSELoss()
        logger.info('Epoch {}/{}'.format(epoch, config.TRAIN.END_EPOCH - 1))
        logger.info('-' * 10)
        acc = utils.AverageMeter()
        acc_classification = utils.AverageMeter()
        batch_loss = utils.AverageMeter()
        class_loss = utils.AverageMeter()
        heatmap_loss = utils.AverageMeter()
        running_loss = 0.0
        running_corrects = 0
        num = 0
        class_2 = 0
        class_1 = 0
        class_0 = 0
        class_3 = 0
        for i, (inputs, labels,class_id) in enumerate(trainloader):
            for id in class_id:
                if id == 2:
                    class_2 = class_2+1
                if id ==1:
                    class_1 = class_1 +1
                if id == 0:
                    class_0 = class_0+1
                if id ==3:
                    class_3 = class_3 +1

            # print(class_id)
            if GradNorm == True:

                inputs, labels, class_id, Weightloss1,Weightloss2 = inputs.to(device), labels.to(device), class_id.to(device), Weightloss1.to(device),Weightloss2.to(device)
                params = torch.tensor(params, device='cuda')
            else:
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

            if GradNorm == True:
                loss_classification = params[0]*criterion_classification(multiclass, class_id)
                loss_heatmap = params[1]*criterion_heatmap(logps, labels.float())
                loss = torch.div(torch.add(loss_classification,loss_heatmap ), 2)
                if epoch==0:

                    l01 = loss_classification.data
                    l02 = loss_heatmap.data

            else:
                loss_classification = criterion_classification(multiclass,class_id)
                loss_heatmap = criterion_heatmap(logps, labels.float())

                loss = loss_classification+config.TRAIN.loss_alpha*loss_heatmap

            batch_loss.update(loss.item(),inputs.size(0))
            class_loss.update(loss_classification.item(),inputs.size(0))
            heatmap_loss.update(loss_heatmap.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if GradNorm == True:
                # Getting gradients of the first layers of each tower and calculate their l2-norm
                param = list(model.parameters())
                G1R = torch.autograd.grad(loss_classification, param[0], retain_graph=True, create_graph=True)
                # print("param[0]",param[0])
                G1 = torch.norm(G1R[0], 2)
                G2R = torch.autograd.grad(loss_heatmap, param[0], retain_graph=True, create_graph=True)
                G2 = torch.norm(G2R[0], 2)
                G_avg = torch.div(torch.add(G1, G2), 2)

                # Calculating relative losses
                lhat1 = torch.div(loss_classification, l01)
                lhat2 = torch.div(loss_heatmap, l02)
                lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)

                # Calculating relative inverse training rates for tasks
                inv_rate1 = torch.div(lhat1, lhat_avg)
                inv_rate2 = torch.div(lhat2, lhat_avg)

                # Calculating the constant target for Eq. 2 in the GradNorm paper
                C1 = G_avg * (inv_rate1) ** alph
                C2 = G_avg * (inv_rate2) ** alph
                C1 = C1.detach()
                C2 = C2.detach()

                opt2.zero_grad()
                # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
                Lgrad = torch.add(Gradloss(G1, C1), Gradloss(G2, C2))
                Lgrad.backward(retain_graph=True)

                # Updating loss weights
                opt2.step()

            optimizer.step()

            if GradNorm == True:
                # Renormalizing the losses weights
                coef = 2 / torch.add(Weightloss1, Weightloss2)
                params = [coef * Weightloss1, coef * Weightloss2]
                # print("Weights are:",Weightloss1, Weightloss2)
                # print("params are:", params)

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
        if GradNorm == True:
            print("Weights are:", Weightloss1, Weightloss2)
            print("params are:", params)

        epoch_loss = running_loss / (num)
        epoch_acc_classification = running_corrects.double() / (num)
        epoch_time = time.time() - initial_time

        writer.add_scalar('Accuracy/heatmap_train', float(acc.avg), epoch)
        writer.add_scalar('Accuracy/class_train', epoch_acc_classification, epoch)
        writer.add_scalar('Loss/total_train', float(batch_loss.avg), epoch)
        writer.add_scalar('Loss_class/train', float(class_loss.avg), epoch)
        writer.add_scalar('Loss_heatmap/train', float(heatmap_loss.avg), epoch)


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

