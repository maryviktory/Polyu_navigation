from PIL import Image
import numpy as np
# load the image
image_PWH = Image.open('/media/maryviktory/My Passport/spine navigation Polyu 2021/DATASET_polyu/FCN_PWH_train_dataset_heatmaps/data_19subj_2/test/labels/sweep018132.png')
# convert image to numpy array
data = np.asarray(image_PWH)
print(type(data))
# summarize shape
print(data.shape)
print(np.argmax(image_PWH))

image_IFL = Image.open('/media/maryviktory/My Passport/IROS 2020 TUM/Spine navigation vertebrae tracking/FCN_spine_point_regression/Heatmaps_spine_train_data/test/labels/Ardit_F100575.png')
# convert image to numpy array
data_IFL = np.asarray(image_IFL)
print(type(data_IFL))
# summarize shape
print(data_IFL.shape)
print(np.argmax(image_IFL))

