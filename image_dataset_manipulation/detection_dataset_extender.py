'''
Image Detection Dataset Extender

Script takes all images as input and multiplies them by changing their contrast, sharpness and brightness values.
By adding changing multipliying factors, dataset can be extended more.

Required data directory structure:

    -data_dir
        |
        |_images
        |    |
        |    |_image1.png
        |    |_image2.png
        |
        |_labels
             |
             |_label1.txt
             |_label2.txt


at the end of the process, data directory will include images, labels, enhanced_images and enhanced_labels

example usage: python dataset_extender.py --data_dir /Users/[data_dir_path]
'''

import os
import shutil
import argparse
from PIL import Image, ImageEnhance


def main(data_dir):
    images_path = os.path.join(data_dir, 'images')
    labels_path = os.path.join(data_dir, 'labels')
    enhanced_image_path = os.path.join(data_dir, 'enhanced_images')
    enhanced_label_path = os.path.join(data_dir, 'enhanced_labels')

    os.makedirs(enhanced_image_path, exist_ok=True)
    os.makedirs(enhanced_label_path, exist_ok=True)

    def image_extender(image_name, factor,type, id_arr):

        factored_image_no = image_name + 1 + id_arr
        factored_image_name = str(factored_image_no) + ".png"
        final_image_name = factored_image_name.zfill(10)
        image_path = os.path.join(enhanced_image_path, final_image_name)
        print("ORIGINAL IMAGE NAME", image_path)

        if type == 'contrast':
            im_output = enhancer.enhance(factor)
        elif type == 'sharp':
            im_output = sharp.enhance(factor)
        elif type == 'bright':
            im_output = bright.enhance(factor)

        im_output.save(image_path)


    def label_extender(label_name, label_path, id_arr):
        factor_label_no = label_name + 1 + id_arr
        factor_label_name = str(factor_label_no) + ".txt"
        final_label_name = factor_label_name.zfill(10)
        final_label_path = os.path.join(enhanced_label_path, final_label_name)
        shutil.copyfile(label_path, final_label_path)

    #Image Extender Part
    len_of_dataset = len(os.listdir(images_path))
    for image in os.listdir(images_path):
        if ".DS_Store" not in image:
            image_path = os.path.join(images_path, image)
            image_name = image.split(".")
            image_name = int(image_name[0])

            im = Image.open(image_path)
            enhancer = ImageEnhance.Contrast(im)
            bright = ImageEnhance.Brightness(im)
            sharp = ImageEnhance.Sharpness(im)

            image_extender(image_name, factor=1, type='contrast', id_arr=0)
            image_extender(image_name, factor=0.8,type='contrast', id_arr=len_of_dataset)
            image_extender(image_name, factor=2, type='sharp', id_arr=len_of_dataset*2)
            image_extender(image_name, factor=1.4, type='bright', id_arr=len_of_dataset*3)
            #if you add more extender, please change the range variable in the second for loop of label extender part.

    #Label Extender Part
    for label in os.listdir(labels_path):
        if ".DS_Store" not in label:
            label_path = os.path.join(labels_path, label)
            label_name = label.split(".")
            label_name = int(label_name[0])

            #since we have 4 factor between the line 80 and 83,
            #we repeat the below process 4 times to match image and label names
            for i in range(4):
                label_extender(label_name, label_path, id_arr=len_of_dataset*i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    main(args.data_dir)

