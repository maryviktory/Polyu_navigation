# from CNN_spine.rospine_utils import *
from torchvision import transforms
import torch
import os
from PIL import Image
import logging
import argparse
import FCN_multiclass.sp_utils as utils
from FCN_multiclass.sp_utils.config import config

num_classes = 4

def main(polyaxon):
    if polyaxon == "True":
        from polyaxon_client.tracking import Experiment, get_data_paths, get_outputs_path

        data_paths = get_data_paths()
        outputs_path = get_outputs_path()
        data_dir = os.path.join(data_paths['data1'], "SpinousProcessData")
        test_dir = os.path.join(data_dir, "PolyU_dataset", "data19subj", "test")

        # model_path = os.path.join(data_paths['outputs1'], "maryviktory", "CNN_spine",
        #                           "groups", "263", "4090",
        #                           "model_best_resnet_fixed_False_pretrained_True_PolyU_dataset.pt")
        model_path = os.path.join(data_dir, "PolyU_dataset", "model_best_resnet_fixed_False_pretrained_True_PolyU_dataset.pt")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.FileHandler(os.path.join(outputs_path, 'SpinousProcess.log')))

        # Polyaxon
        experiment = Experiment()
        logger.info("model_dir {}".format(model_path))

    else:
        # test_dir = "/media/maryviktory/My Passport/IPCAI 2020 TUM/CNN/data_all(15patients train, 4 test)/test"
        test_dir = "D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps\data_19subj_multiclass_heatmap\\test"
        model_path = "/media/maryviktory/My Passport/IROS 2020 TUM/DATASETs/Dataset/PWH_sweeps/models/model_best_resnet_fixed_False_pretrained_True_data_19subj_2_exp_36776.pt"

    # print(polyaxon)
    # print("model path: ",model_path)
    print("test dir: ",test_dir)

    # model = ModelLoader_types(num_classes, model_path, model_type="classification")
    model = utils.model_pose_resnet.get_pose_net(config.TEST.Windows_MODEL_FILE, is_train=False)
    print('=> loading model from {}'.format(config.TEST.Windows_MODEL_FILE))
    model.load_state_dict(
        torch.load(config.TEST.Windows_MODEL_FILE, map_location=torch.device('cpu'))['model_state_dict'])

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
                image_batch, _ = utils.list2batch([tensor_image], None)

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

        print('correct', n_correct, n_correct / n_total)
        print("true positive", n_tp, n_tp / n_total)
        print("false positive", n_fp, n_fp / n_total)
        print("false negative", n_fn, n_fn / n_total)
        print("true negative", n_tn, n_tn / n_total)
        print('total', n_total)

    if num_classes == 3:

        gap_list = os.listdir(os.path.join(test_dir, "1Sacrum"))
        non_gap_list = os.listdir(os.path.join(test_dir, "2Lumbar"))

        gap_list = [os.path.join(test_dir, "1Sacrum", item) for item in gap_list]
        non_gap_list = [os.path.join(test_dir, "2Lumbar", item) for item in non_gap_list]

        sacrum_list = os.listdir(os.path.join(test_dir, "3Thoracic"))
        sacrum_list = [os.path.join(test_dir, "3Thoracic", item) for item in sacrum_list]

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
                image_batch, _ = utils.list2batch([tensor_image], None)

                logps,out = model.forward(image_batch)
                prob = torch.sigmoid(out).detach().numpy()
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

        if polyaxon == "True":
            logger.info('correct {} ({})'.format(n_correct, n_correct/n_total))
            logger.info("true gap predicted spinous {} ({})".format( n_gap_sp,n_gap_sp/n_total))
            logger.info("true gap predicted sacrum {} ({})".format(n_gap_sacrum, n_gap_sacrum / n_total))
            logger.info("true spinous predicted gap {} ({})".format( n_sp_gap, n_sp_gap / n_total))
            logger.info("true spinous predicted sacrum {} ({})".format(n_sp_sacrum, n_sp_sacrum / n_total))
            logger.info("true sacrum predicted gap {} ({})".format(n_sacrum_gap, n_sacrum_gap / n_total))
            logger.info("true sacrum predicted spinous {} ({})".format(n_sacrum_sp, n_sacrum_sp / n_total))
            logger.info('total {}'.format(n_total))
        else:
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
                image_batch, _ = utils.list2batch([tensor_image], None)

                _,out = model.forward(image_batch)
                prob = torch.sigmoid(out).detach().numpy()
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

        if polyaxon == "True":
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')

    parser.add_argument('--flag_Polyaxon', type=str, default="False",
                        help='TRUE - Polyaxon, False - CPU')

    args = parser.parse_args()

    polyaxon = args.flag_Polyaxon
    main(polyaxon)
