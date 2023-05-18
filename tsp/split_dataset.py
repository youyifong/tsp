import numpy
import os
from PIL import Image


def split_dataset_by_class(DATASET_FOLDER_NAME, IMG_EXT='tif', TRAIN_RATIO = 0.9):
    class_names = os.listdir(DATASET_FOLDER_NAME)
    test_path, train_path = prepare_split_target_folders()
    print(f"IMG_EXT: {IMG_EXT}, TRAIN_RATIO: {TRAIN_RATIO}")

    for class_name in class_names:
        current_class_path = os.path.join(DATASET_FOLDER_NAME, class_name)
        current_class_images = list((filter(lambda filename: filename.endswith(IMG_EXT), os.listdir(current_class_path))))
        number_of_images_in_class = len(current_class_images)
        print(f"{class_name} number_of_images_in_class: {number_of_images_in_class}")

        sampling_permuatation = numpy.random.permutation(number_of_images_in_class)
        train_index_bound = int(TRAIN_RATIO * number_of_images_in_class)

        category_test_path, category_train_path = prepare_class_folder(class_name, test_path, train_path)

        for current_index in range(train_index_bound):
            copy_image_to_destination_folder(current_class_images,
                                             sampling_permuatation,
                                             current_index,
                                             current_class_path,
                                             category_train_path)

        for current_index in range(train_index_bound, number_of_images_in_class):
            copy_image_to_destination_folder(current_class_images,
                                             sampling_permuatation,
                                             current_index,
                                             current_class_path,
                                             category_test_path)


def copy_image_to_destination_folder(current_class_images, sampling_permutation, current_index, current_class_path, destination_path):
    current_image_name = current_class_images[sampling_permutation[current_index]]
    current_image_path = os.path.join(current_class_path, current_image_name)

    current_image_dst_path = os.path.join(destination_path, current_image_name).replace('.tif', '.jpg')

    image = Image.open(current_image_path)
    image.save(current_image_dst_path)


def prepare_class_folder(class_name, test_path, train_path):
    train_path_for_class = os.path.join(train_path, class_name)
    test_path_for_class = os.path.join(test_path, class_name)
    custom_mkdir(train_path_for_class)
    custom_mkdir(test_path_for_class)
    return test_path_for_class, train_path_for_class


def prepare_split_target_folders():
    train_path = os.path.join('./', "train")
    test_path = os.path.join('./', "test")
    custom_mkdir('train')
    custom_mkdir('test')
    return test_path, train_path


def custom_mkdir(train_path):
    if not os.path.exists(train_path):
        os.mkdir(train_path)


