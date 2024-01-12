from melee_analytics_platform.pascal_voc_xml_modeling.model import Model
from melee_analytics_platform.pascal_voc_xml_modeling.data_utils import train_test_split, write_json
from melee_analytics_platform.pascal_voc_xml_modeling.datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from datetime import datetime
import os


def train_n_test(project_name,
                 batch_size = 6,
                 resize_width = 416,
                 resize_height = 416,
                 epochs = 5,
                 num_workers = 2,
                 show_image = False,
                 shuffle = True,
                 train_percent = .8,
                 overall_sample = 1,
                 min_labels = 25,
                 balance_classes = True,
                 detection_threshold = .8
                 ):
    """
    This is the one function to rules them all. It creates a model using a set of images
    and annotations saved off in '/projects/[project_name]/images/' and
    '/projects/[project_name]/annotations/'. Everything is saved in a subdirectory of in
    '/projects/[project_name]/outputs/' in a directory named be the date and time of the
    run to avoid overwriting and promote iteration and testing. It saves the model, a json
    of parameters, validation and testing loss charts, and validation image outputs in the
    subdirectory. Tons of parameters to control

    Parameters:
    project_name: str
        The name of the project directory to pull training data from and to save outputs to
        
    batch_size: int
        Number of images to batch to the device at once, default 6
        
    resize_width: int
        width to resize training images to, default 416
        
    resize_height: int
        height to resize images to, default 416
        
    epochs: int
        number of epochs to use in training before validation steps, default 5
        
    num_workers: int
        number of workers to use in training, default 2

    shuffle: bool
        shuffle training data before it's iterated over and used for training, default True

    train_percent: float
        the percent of total training data to use in training vs validation, default .8

    overall_sample: float
        the overall percent sample of training data to be taken before splitting. Great for testing quickly, default 1

    min_labels: int
        The minimum number of labels a detection class must have to be used for modeling, default 25

    balance_classes: bool
        Attempt to remove images/annotations if they only contain labels for overbalanced classes, default True

    detection_threshold: float
        The output score percent threshold to trigger a positive class detection in validation images, default .8
    """
    # Get the current date and time to use in the model directory later
    project_run = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Starting model in directory {project_run}\n')

    # make the output path if it doesn't exist
    if not os.path.exists(f"projects/{project_name}/outputs/{project_run}"):
        os.makedirs(f"projects/{project_name}/outputs/{project_run}")

    # get our train and test image lists, as well as the classes we'll be using
    train_list, test_list, classes = train_test_split(project_name,
                                                      train_percent,
                                                      overall_sample,
                                                      min_labels = min_labels,
                                                      balance_classes = balance_classes)    

    # make a big ole json load file
    json_load = {'project_name': project_name,
                 'project_run': project_run,
                 'classes': classes,
                 'overall_sample': overall_sample,
                 'train_percent': train_percent,
                 'epochs': epochs,
                 'resize_width': resize_width,
                 'resize_height': resize_height,
                 'shuffle': shuffle,
                 'batch_size': batch_size,
                 'num_workers': num_workers,
                 'min_labels' : min_labels,
                 'balance_classes' : balance_classes,
                 'train_list': train_list,
                 'test_list': test_list}

    # slap this all into a nice new json to reference for later
    write_json(f'projects/{project_name}/outputs/{project_run}', json_load, 'run_params')

    # take our list of training data and make tensor dataset and loader for it
    train_dataset = create_train_dataset(project_name, project_run, train_list, resize_width, resize_height)
    train_loader = create_train_loader(train_dataset,
                                       num_workers = num_workers,
                                       batch_size = batch_size,
                                       shuffle = shuffle)
    print(f"Number of training samples: {len(train_dataset)}")

    # create a model
    model = Model(project_name=project_name, project_dir=project_run)

    # and train!
    model.train(train_loader)

    # visual confirmation of what its doing to the test images
    model.predict_on_images(detection_threshold = detection_threshold)

    # now get the validation loss
    test_dataset = create_valid_dataset(project_name, project_run, train_list, resize_width, resize_height)
    test_loader = create_valid_loader(test_dataset,
                                       num_workers = num_workers,
                                       batch_size = batch_size,
                                       shuffle = shuffle)
    model.train(test_loader, test=True)