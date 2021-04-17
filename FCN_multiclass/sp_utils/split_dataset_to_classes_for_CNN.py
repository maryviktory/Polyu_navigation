import cv2
import pandas as pd
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--dir', default="D:\spine navigation Polyu 2021\DATASET_polyu\PWH_sweeps\Subjects dataset\\new", help='Database directory')


def main(root_dir):

    for sublect_folder in os.listdir(root_dir):
        print(sublect_folder)
        pd_frame = pd.DataFrame(columns=["image_path", "image_class"])
        labels_dir = os.path.join(os.path.join(root_dir,sublect_folder, "Labels"))
        image_dir = os.path.join(os.path.join(root_dir, sublect_folder,"Images"))
        images_list = os.listdir(labels_dir)
        print(images_list)
        for item in images_list:
            label = cv2.imread(os.path.join(labels_dir, item), 0)
            image = cv2.imread(os.path.join(image_dir, item), 0)
            image_class = int(np.sum(label) > 0)
            pd_frame = pd_frame.append({'image_path': os.path.join("Images", item), 'image_class': image_class},
                                       ignore_index=True)

            if not os.path.exists(os.path.join(root_dir,sublect_folder, "Classes", "NonGap")):
                os.makedirs(os.path.join(root_dir,sublect_folder, "Classes", "NonGap"))

            if not os.path.exists(os.path.join(root_dir,sublect_folder, "Classes", "Gap")):
                os.makedirs(os.path.join(root_dir,sublect_folder, "Classes", "Gap"))

            if image_class == 1:

                cv2.imwrite(os.path.join(root_dir,sublect_folder, "Classes", "NonGap", item), image)
            else:
                cv2.imwrite(os.path.join(root_dir,sublect_folder, "Classes", "Gap", item), image)

        pd_frame.to_csv(os.path.join(root_dir,sublect_folder, "data_frame.csv"))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.dir)