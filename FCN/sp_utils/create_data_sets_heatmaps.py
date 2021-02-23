
import argparse
import os
import shutil
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--dir', default="D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_dataset_heatmaps_all", help='Database directory')
parser.add_argument('--outdir', default="D:\spine navigation Polyu 2021\DATASET_polyu\FCN_PWH_train_dataset_heatmaps", help='Database directory')

args = parser.parse_args()


def create_folder(folder,data_set,mode,clas):
    # folder = "/media/maria/My Passport/toNas/"
    data_folder= os.path.join(folder,'%s'%data_set)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    mode_folder= os.path.join(data_folder, "%s"%(mode))
    if not os.path.exists(mode_folder):
        os.mkdir(mode_folder)

    destination_folder = os.path.join(mode_folder, "%s" % (clas))
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    return destination_folder


def open_file(name_file):
    train_file = open(name_file, "r")
    lines_train = train_file.read().splitlines()
    print('patient names:',lines_train)
    return lines_train

# train_file = open_file("dataset_run5_training.txt")
# val_file = open_file("dataset_run5_validation.txt")
# test_file = open_file('dataset_run_test.txt')

train_list = ["sweep017",
"sweep5007",
"sweep3006",
'sweep012',
"sweep013",
"sweep014",
"sweep015",
"sweep019",
"sweep020",
"sweep3005",
"sweep5001",
"sweep3001"
]
val_list = ["sweep3000",
"sweep9004",
"sweep10001"
]
test_list = ["sweep018","sweep3013","sweep5005", "sweep9001"]

train_list =["phantom",
"phantom_sweep_3","phantom_sweep_7",
"phantom_sweep_8"]
test_list = ["phantom_sweep_4"]
val_list = ["phantom_sweep_6"]

def copy_files (data_set, file,mode):

    for i in file:

        image_subj_folder = os.path.join(os.path.join(args.dir, "%s"%i,'Images'))
        label_subj_folder = os.path.join(os.path.join(args.dir, "%s" % i, 'Labels'))


        image_dataset_folder = create_folder(args.outdir,'%s'%data_set, '%s'%mode, 'images')
        labels_dataset_folder = create_folder(args.outdir, '%s' % data_set, '%s' % mode, 'labels')


        for filename in os.listdir(image_subj_folder):
            full_file_name = os.path.join(image_subj_folder, filename)
            label_file_name = os.path.join(label_subj_folder,filename)
            # print(np.sum(label_file_name))
            label = cv2.imread(label_file_name, 0)
            # print(label)
            if int(np.sum(label) > 0): #os.path.isfile(full_file_name) and
                shutil.copy(full_file_name, image_dataset_folder)
                shutil.copy(label_file_name, labels_dataset_folder)
            # print('filenames in Gap',filename)


# copy_files('phantom_6scans',test_list,'test')
# copy_files('data_19subj_2',train_list,'train')
# copy_files('data_19subj_2',val_list,'validation')
# copy_files('data_19subj_2',test_list,'test')