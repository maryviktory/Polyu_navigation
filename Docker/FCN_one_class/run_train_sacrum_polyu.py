import sp_utils as utils
# from sp_utils.config import config
from sp_utils.config import config
import argparse
import logging
import os
import stat
from datetime import datetime
# from torch_lr_finder import LRFinder
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

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels.float())
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

        writer.add_scalar('Loss/val', float(val_loss / len(valloader)), epoch)
        # writer.add_scalar('Accuracy/val', float(accuracy / len(valloader)), epoch)
        writer.add_scalar('Accuracy/val', acc.avg, epoch)


    return acc.avg

def main(config):

    logging.basicConfig(level=logging.INFO)
    logging.info("STARTING PROGRAM")

    # datetime object containing current date and time
    now = datetime.now()

    print("now =", now)

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y %H_%M_%S")
    # print("date and time =", dt_string)

    config.DATASET.OUTPUT_PATH = os.path.join("runs", '%s' % dt_string)
    print("output path", config.DATASET.OUTPUT_PATH)
    if not os.path.exists(config.DATASET.OUTPUT_PATH):
        os.makedirs(config.DATASET.OUTPUT_PATH)

    os.chmod(config.DATASET.OUTPUT_PATH, stat.S_IRWXO)
    os.chmod(os.path.join("runs"), stat.S_IRWXO)
    # os.chmod(config.DATASET.PATH, stat.S_IRWXO)
    # os.chmod("data_24subj_sacrum_heatmap",stat.S_IRWXO)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(config.DATASET.OUTPUT_PATH , 'Heatmaps_Resnet101.log')))
    model_path = config.MODEL.PRETRAINED

    trainloader, valloader = utils.load_split_train_val(config.DATASET.PATH, "train", "validation", config)


    logger.info('batch size {}'.format(config.TRAIN.BATCH_SIZE))
    logger.info('dataset {}'.format( config.DATASET.PATH))
    logger.info("weights {}".format(config.TRAIN.UPDATE_WEIGHTS))
    # logger.info("weights loss {}".format(config.TRAIN.loss_alpha))
    # logger.info("Model: {}".format(model_path))
    logger.info("LR: {}".format(config.TRAIN.LR))
    logger.info("Imagenet: {}".format(config.MODEL.Imagenet_pretrained))


    model = utils.model_pose_resnet.get_pose_net(config,model_path,is_train = True)

    model.eval()

    for name,parameter in model.named_parameters():
        parameter.requires_grad = config.TRAIN.UPDATE_WEIGHTS
        if "deconv" in name or "final" in name:
            parameter.requires_grad = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    model.to(device)



    #NOTE: uncomment for LR finder
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)

    # lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    # print("finding range")
    # lr_finder.range_test(trainloader,val_loader=valloader, end_lr=100, num_iter=100)
    # lr_finder.plot()  # to inspect the loss-learning rate graph
    # lr_finder.reset()  # to reset the model and optimizer to their initial state

    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.01)

    writer = SummaryWriter(config.DATASET.OUTPUT_PATH)
    best_acc = 0

    for epoch in range(config.TRAIN.END_EPOCH):
        criterion = nn.MSELoss()
        logger.info('Epoch {}/{}'.format(epoch, config.TRAIN.END_EPOCH - 1))
        logger.info('-' * 10)
        acc = utils.AverageMeter()
        batch_loss = utils.AverageMeter()

        for i, (inputs, labels) in enumerate(trainloader):

            inputs, labels = inputs.to(device), labels.to(device)

            # print(summary(model, tuple(inputs.size())[1:]))
            logps = model.forward(inputs)

            criterion = nn.MSELoss()
            loss = criterion(logps, labels.float())
            batch_loss.update(loss.item(),inputs.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, avg_acc, cnt, pred,target,dists = utils.accuracy(logps.detach().cpu().numpy(),
                                             labels.detach().cpu().numpy(),thr = config.TRAIN.THRESHOLD)
            # print("Current batch accuracy: ", avg_acc)
            acc.update(avg_acc,cnt)
            # print("Batch {} train accurcy: {}, loss: {}".format(i, acc.avg, batch_loss.avg))
        writer.add_scalar('Loss/train', float(batch_loss.avg), epoch)


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

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': batch_loss.avg,
            }, os.path.join(config.DATASET.OUTPUT_PATH, "model" + str(epoch) + ".pt"))

    logger.info('Best val Acc: {:4f}'.format(best_acc))

def Parser():
    parser = argparse.ArgumentParser(description='DeepSpine script')

    parser.add_argument('--data_dir', type=str, default="data_24subj_sacrum_heatmap", metavar='N',
                        help='')

    parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--update_weights', type=bool, default=True, metavar='P',
                        help='whether to train the networs from scratches or with fine tuning')

    parser.add_argument('--lr', type=float, default=0.001, metavar='BS',
                        help='learning rate')
    args = parser.parse_args()

    return args

def update_config(config,args):
    if args.data_dir:
        config.DATASET.PATH = args.data_dir
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