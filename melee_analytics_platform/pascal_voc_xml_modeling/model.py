from melee_analytics_platform.pascal_voc_xml_modeling.custom_utils import Averager, save_model, save_loss_plot
from melee_analytics_platform.pascal_voc_xml_modeling.data_utils import read_json, update_json
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import cv2
import torch
import glob as glob
from tqdm.auto import tqdm
import os
import time


class Model():
    def __init__(self, project_name, project_dir):
        # set the project name
        self.project_name = project_name

        # handle project directory
        self.project_dir = project_dir

        # read in the params from the json and assign
        self.project_json = read_json(f'projects/{self.project_name}/outputs/{self.project_dir}', 'run_params')
        self.classes = self.project_json['classes']
        self.epochs = self.project_json['epochs']

        # assign the device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # make a model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # get the number of input features 
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes))
        # see if a model checkpoint exists
        if os.path.isfile(f'projects/{self.project_name}/outputs/{self.project_dir}/model.pth'):
            print('loading model state')
            checkpoint = torch.load(f'projects/{self.project_name}/outputs/{self.project_dir}/model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # and send it to the device
        self.model.to(self.device)

    def train(self, loader, test = False):
        # switch this guy over to training mode if necessary
        if not self.model.training:
            self.model.train()

        # check if we're training or testing
        if test:
            mode = 'test'
            epochs = 1
        else:
            mode = 'train'
            epochs = self.epochs

        # get the model parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        # define the optimizer
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        # initialize the Averager class
        loss_hist = Averager()
        # train and validation loss lists to store loss values of all...
        # ... iterations till ena and plot graphs for all iterations
        loss_list = []

        # start the training epochs
        for epoch in range(epochs):
            print(f"\nEPOCH {epoch+1} of {epochs}")
            # reset the training and validation loss histories for the current epoch
            loss_hist.reset()
            # start timer and carry out training and validation
            start = time.time()

            ### TRAINING ###
            # initialize tqdm progress bar
            prog_bar = tqdm(loader, total=len(loader))
            
            for i, data in enumerate(prog_bar):
                optimizer.zero_grad()
                images, targets = data
                
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                loss_list.append(loss_value)
                loss_hist.send(loss_value)
                losses.backward()
                optimizer.step()

                # update the loss value beside the progress bar for each iteration
                prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
            total_loss = loss_list

            # print out how we did
            print(f"Epoch #{epoch+1} {mode} loss: {loss_hist.value:.3f}")   
            
            end = time.time()
            # also print a nice time update
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")
            
            # save loss plot from training
            save_loss_plot(f"projects/{self.project_name}/outputs/{self.project_dir}", total_loss, mode)
            
            # sleep for 3 seconds after each epoch
            time.sleep(3)
        # add the training losses to the json
        update_json(f'projects/{self.project_name}/outputs/{self.project_dir}', {f'{mode}ing_loss': loss_list}, 'run_params')
        # save out the model
        save_model(f'projects/{self.project_name}/outputs/{self.project_dir}', epoch, self.model, optimizer)

    def predict(self,
                image,
                detection_threshold = .9):
        """
        Given an image and project name, make some predictions and return it.
        image is already a lil numpy thing
        """
        print(detection_threshold)
        # check if we need to switch to eval mode
        if self.model.training:
            self.model.eval()

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        cv2.imwrite('my_image.jpg', image)
        
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = self.model(image.to(self.device))
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            # get all the predicited class names
            pred_classes = np.array([self.classes[i] for i in outputs[0]['labels'].cpu().numpy()])
            pred_classes = pred_classes[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]

            return boxes, scores, pred_classes
        else:
            return [], [], []
    

    def predict_on_images(self,
                          detection_threshold = .9):
        """
        Given a project name and a list of images, predict on all of the images.
        Will write output predictions in outputs/validation
        """    
        # read in the run params
        test_images = self.project_json['test_list']

        # this will help us create a different color for each class
        COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # make the output path if it doesn't exist
        if not os.path.exists(f"projects/{self.project_name}/outputs/{self.project_dir}/validation_images"):
            os.makedirs(f"projects/{self.project_name}/outputs/{self.project_dir}/validation_images")

        # lil print helper before we iterate
        print(f'\nRunning Predictions on {len(test_images)} Test Images...')
        for i in range(len(test_images)):
            # get the image file name for saving output later on
            image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
            image = cv2.imread(f'projects/{self.project_name}/images/{test_images[i]}')
            # prediction time
            boxes, scores, pred_classes = self.predict(image.copy(), detection_threshold = detection_threshold)
            if len(boxes) != 0:            
                # draw the bounding boxes and write the class name on top of it
                for j, box in enumerate(boxes):
                    class_name = pred_classes[j]
                    color = COLORS[self.classes.index(class_name)]
                    cv2.rectangle(image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                color, 2)
                    cv2.putText(image, f'{class_name}: {round(float(scores[j]), 2)}', 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                                2, lineType=cv2.LINE_AA)
                cv2.imwrite(f'projects/{self.project_name}/outputs/{self.project_dir}/validation_images/{image_name}.jpg', image)
        print('TEST PREDICTIONS COMPLETE')