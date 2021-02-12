'''
Script to extend image classification dataset size by altering Brightness, Contrast and Sharpness of images.
It deletes all classes that has an image examples lower than the "min_number_of_examples" variable.

Expected data directory structure is:

    -data_dir
        |
        |_train_dir
        |    |
        |    |_class1(folder)
        |    |   |
        |    |   |_images
        |    |
        |    |_class2(folder)
        |        |
        |        |_images
        |
        |_validation_dir
        |    |
        |    |_class1(folder)
        |    |   |
        |    |   |_images
        |    |
        |    |_class2(folder)
        |        |
        |        |_images
        |
        |_test_dir
             |
             |_class1(folder)
             |   |
             |   |_images
             |
             |_class2(folder)
                 |
                 |_images

'''
import os
import shutil
import random
import argparse

from PIL import Image, ImageEnhance


def main(data_dir, min_number_of_examples):

    image_new_name_ext = 0
    image_numbers = []
    label_order = []
    image_new_name_extention = 0
    object_in_class = 0
    
    train = os.path.join(data_dir, 'train_dir')
    test = os.path.join(data_dir, 'test_dir')
    validation = os.path.join(data_dir, 'validation_dir')
    os.makedirs(validation,  exist_ok = True)
    os.makedirs(test, exist_ok = True)
    
    def get_info(train, object_in_class):
        for label in os.listdir(train):
            if '.DS_Store' not in label:
                label_paths = os.path.join(train, label)
                label_order.append(label_paths)
                image_numbers.append(len(os.listdir(label_paths)))
                object_in_class+=len(os.listdir(label_paths))
        print("Number of classes in dataset:", len(os.listdir(train)))
        print("Dataset Size:",object_in_class)
        print("=======================")

    get_info(train,object_in_class)

    def multiplier(image_new_name_extention, image_numbers,label_order, max_number, rotation, enh, b_factor):
        for i, path in zip(image_numbers,label_order):
            extended_value = max_number-i
            if len(os.listdir(path)) <= max_number*0.65:
                for image in os.listdir(path):
                    if '.DS_Store' not in image:
                        image_new_name_extention+=1
                        image_path = os.path.join(path, image)
                        im = Image.open(image_path)
                        image_name = image.split(".")
                        image_name_ = image_name[0] + str(image_new_name_extention) +'.jpg'
                        new_image_path = path + "/" + image_name_

                        if enh=="contrast":
                            enhancer = ImageEnhance.Contrast(im)
                            b_factor = b_factor
                        elif enh =="bright":
                            enhancer = ImageEnhance.Brightness(im)
                            b_factor = b_factor
                        elif enh =="sharp":
                            enhancer = ImageEnhance.Sharpness(im)
                            b_factor = b_factor

                        im_output2 = enhancer.enhance(b_factor)
                        im_output = im_output2.rotate(rotation)
                        im_output.save(new_image_path)
                        if len(os.listdir(path)) > max_number:
                            break


    max_number = max(image_numbers)
    min_number = min(image_numbers)

    for i, path in zip(image_numbers,label_order):
        extended_value = max_number-i
        if i<min_number_of_examples:
            print(i, path)
            label_order.remove(path)
            image_numbers.remove(i)
            shutil.rmtree(path)

    print("max example:", max_number, "min example:", min_number)


    multiplier(image_new_name_extention, image_numbers,label_order, max_number,rotation=1, enh = 'bright', b_factor=1.3)
    get_info(train,object_in_class)
    multiplier(image_new_name_extention, image_numbers,label_order, max_number,rotation=-1 , enh = 'contrast', b_factor=1.1)
    get_info(train,object_in_class)
    multiplier(image_new_name_extention, image_numbers,label_order, max_number, rotation=0.5, enh = 'sharp', b_factor=1.2)
    get_info(train,object_in_class)
    
    
    
    shuffle_test_list = []


    for label in os.listdir(train):
        if '.DS_Store' not in label:
            label_path = os.path.join(train, label)
            for index, image in enumerate(os.listdir(label_path)):
                image_path = os.path.join(label_path, image)
                test_label_path = os.path.join(test,label)
                shuffle_test_list.append(image_path)
                os.makedirs(test_label_path,  exist_ok = True)
                
                
    random.shuffle(shuffle_test_list)
    for index, image in enumerate(shuffle_test_list):
        image_new_path = image.replace("train_dir", "test_dir")
        if index<=int(len(shuffle_test_list)*0.1):
            dest = shutil.move(image, image_new_path)
            
            

    shuffle_val_list = []

    for label in os.listdir(train):
        if '.DS_Store' not in label:
            label_path = os.path.join(train, label)
            for index, image in enumerate(os.listdir(label_path)):
                image_path = os.path.join(label_path, image)
                val_label_path = os.path.join(validation,label)
                shuffle_val_list.append(image_path)
                os.makedirs(val_label_path,  exist_ok = True)
                
                
    random.shuffle(shuffle_val_list)
    for index, image in enumerate(shuffle_val_list):
        image_new_path = image.replace("train_dir", "validation_dir")
        if index<=int(len(shuffle_val_list)*0.22):
            dest = shutil.move(image, image_new_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--min_number_of_examples', type=int, required=True)

    
    args = parser.parse_args()
    main(args.data_dir, args.min_number_of_examples)
