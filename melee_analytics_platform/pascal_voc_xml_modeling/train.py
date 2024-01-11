from config import DEVICE
from model import create_model, predict_on_images, get_testing_loss
from data_utils import get_classes, train_test_split, write_json, read_json, update_json
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import time
import os
plt.style.use('ggplot')

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
        
    show_image: bool
        display a qa check image before training, will pause training until closed, default False

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
    # create a model and send it to our device of choice
    model = create_model(num_classes=len(classes))
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if show_image:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader, project_name)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel(project_name)
    # start the training epochs
    for epoch in range(epochs):
        print(f"\nEPOCH {epoch+1} of {epochs}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()

        ### TRAINING ###
        # initialize tqdm progress bar
        prog_bar = tqdm(train_loader, total=len(train_loader))
        
        for i, data in enumerate(prog_bar):
            optimizer.zero_grad()
            images, targets = data
            
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)
            losses.backward()
            optimizer.step()

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        train_loss = train_loss_list

        # print out how we did
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
          
        end = time.time()
        # also print a nice time update
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
        
        # save loss plot from training
        save_loss_plot(f"projects/{project_name}/outputs/{project_run}", train_loss, 'train')
        
        # sleep for 5 seconds after each epoch
        time.sleep(5)
    # add the training losses to the json
    update_json(f'projects/{project_name}/outputs/{project_run}', {'training_loss': train_loss_list}, 'run_params')
    # save out the model
    save_model(f'projects/{project_name}/outputs/{project_run}', epoch, model, optimizer)
    ### validation time ### 
    predict_on_images(project_name, project_run, detection_threshold = detection_threshold)
    get_testing_loss(project_name, project_run)