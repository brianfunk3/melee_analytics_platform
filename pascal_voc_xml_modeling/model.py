import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
import torch
import glob as glob
from tqdm.auto import tqdm
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
import os
import time
from datasets import create_valid_dataset, create_valid_loader
from data_utils import get_classes, train_test_split, read_json, get_boxes_labels, update_json
from config import DEVICE

def create_model(num_classes):
    """Create a default FasterCNN model with num_classes number of classes"""
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def get_most_recent_project_run(project_name):
    """
    In the event that a project run isn't provided, we'll default
    provide the most recent one, as a treat
    """
    return max([folder for folder in os.listdir(f'projects/{project_name}/outputs/')])

def get_model(project_name, project_run = None):
    """
    Given a project name, return a model thats ready to PARTY
    """ 
    # check-a-roo
    if project_run is None:
        project_run = get_most_recent_project_run(project_name)
    
    # find classes
    classes = get_classes(project_name, project_run)

    # beyblade
    model = create_model(num_classes=len(classes))
    checkpoint = torch.load(f'projects/{project_name}/outputs/{project_run}/model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    return model

def predict(model,
            classes,
            image,
            detection_threshold = .9):
    """
    Given an image and project name, make some predictions and return it.
    image is already a lil numpy thing
    """
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image.to(DEVICE))
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

        return boxes, scores, pred_classes
    else:
        return [], [], []
    

def predict_on_images(project_name,
                      project_run = None,
                      detection_threshold = .8):
    """
    Given a project name and a list of images, predict on all of the images.
    Will write output predictions in outputs/validation
    """
    # check-a-roo
    if project_run is None:
        project_run = get_most_recent_project_run(project_name)

    # get the model
    model = get_model(project_name, project_run).eval()
    
    # read in the run params
    run_params = read_json(f'projects/{project_name}/outputs/{project_run}', 'run_params')
    classes = run_params['classes']
    test_images = run_params['test_list']

    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # make the output path if it doesn't exist
    if not os.path.exists(f"projects/{project_name}/outputs/{project_run}/validation_images"):
        os.makedirs(f"projects/{project_name}/outputs/{project_run}/validation_images")

    # lil print helper before we iterate
    print(f'\nRunning Predictions on {len(test_images)} Test Images...')
    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(f'projects/{project_name}/images/{test_images[i]}')
        # prediction time
        boxes, scores, pred_classes = predict(model, classes, image.copy(), detection_threshold = detection_threshold)
        if len(boxes) != 0:            
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(boxes):
                class_name = pred_classes[j]
                color = COLORS[classes.index(class_name)]
                cv2.rectangle(image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color, 2)
                cv2.putText(image, f'{class_name}: {round(float(scores[j]), 2)}', 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                            2, lineType=cv2.LINE_AA)
            cv2.imwrite(f'projects/{project_name}/outputs/{project_run}/validation_images/{image_name}.jpg', image)
    print('TEST PREDICTIONS COMPLETE')

def get_testing_loss(project_name, project_run):
    # first we need to get our model and some classes
    model = get_model(project_name, project_run).train()
    # read in the run params
    run_params = read_json(f'projects/{project_name}/outputs/{project_run}', 'run_params')
    classes = run_params['classes']
    test_list = run_params['test_list']
    resize_width = run_params['resize_width']
    resize_height = run_params['resize_height']
    num_workers = run_params['num_workers']
    batch_size = run_params['batch_size']
    shuffle = run_params['shuffle']

    # loader?
    valid_dataset = create_valid_dataset(project_name, project_run, test_list, resize_width, resize_height)
    valid_loader = create_valid_loader(valid_dataset,
                                       num_workers = num_workers,
                                       batch_size = batch_size,
                                       shuffle = shuffle)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    val_loss_hist = Averager()
    val_loss_list = []
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_loader, total=len(valid_loader))
            
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
               
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    val_loss = val_loss_list
    print(f"Validation loss: {val_loss_hist.value:.3f}")
    update_json(f'projects/{project_name}/outputs/{project_run}', {'validation_loss': loss_value}, 'run_params')
    save_loss_plot(f"projects/{project_name}/outputs/{project_run}", val_loss, 'validation', custom_label = f'Validation Loss: {loss_value:.4f}')