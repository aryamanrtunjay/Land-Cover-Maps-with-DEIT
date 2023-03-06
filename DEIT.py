from PIL import Image
import torch
from torch import nn
import timm
import time
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import torchvision
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import wandb
from sklearn.metrics import RocCurveDisplay

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, epoch):
        # Set up correction curve
        dx = -0.65
        denom = 4 ** -3
        min_val = 0.01
        correction_increase = (epoch ** dx) / denom + min_val

        # Get normal loss
        criterion = nn.CrossEntropyLoss().cuda()
        init_loss = criterion(output, target)

        # Initialize new loss value
        loss = init_loss

        # Loop through model predictions and check if Banana was predicted as Maize
        for i in range(len(target)):
            if(target[i] == 2 and torch.argmax(output[i]) == 3):
                # If banana predicted as maize, modify the loss based on number of training steps completed
                loss = init_loss + correction_increase

        return loss

def plot_ROC_curve(target, output, train_ds, positive_class, name):
    # Get the label being set as the positive value
    positive_key = list(train_ds.dataset.class_to_idx.keys())[positive_class]
    # Change target list from correct label to whether label is equal to positive key
    roc_target = [int(x.item() == positive_class) for x in target]
    # Convert output tensor to list
    scores =  [x for x in output]
    # Get the false positive rates, true positive rates, and thresholds
    fpr, tpr, thresholds = metrics.roc_curve(roc_target, scores, pos_label=1)

    # Display ROC Curve
    RocCurveDisplay.from_predictions(
        roc_target,
        scores,
        name=f"{positive_key} vs the rest",
        color="darkorange",
    )

    # Plot ROC of random classifier (50-50 pos neg)
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(positive_key)
    plt.legend()
    plt.savefig(f"ROC_plots/ROC_{name}_{positive_key}.png")
    return (True)

print(torch.__version__)
# should be 1.8.0

# Set seed to get repeatable results
torch.manual_seed(2023)

# Initialize run in weights and biases
wandb.init(project="DEIT", name="DEIT-Final")
wandb.config = {
    "learning_rate": 0.000004,
    "epochs": 50,
    "batch_size": 32
}

# Load base DEIT model
model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained = True)
# Modify structure
model.head = torch.nn.Linear(in_features = 768, out_features = 5, bias = True)
model.head_dist = torch.nn.Linear(in_features = 768, out_features = 5, bias = True)
# Optimize model for training
model.train()
# Set up model to run with Computer Unified Device Architecture (CUDA)
model = model.cuda()

# Set up augmentations
transform = transforms.Compose([
    transforms.Resize(224, interpolation=3),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

# Get training data
train_imgs = torchvision.datasets.ImageFolder(
    root = "Split_Data/training",
    transform=transform
)

# Get testing data
test_imgs = torchvision.datasets.ImageFolder(
    root = "Split_Data copy/validation",
    transform=transform
)

batch_size = 32
# Convrt ImageFolder object to DataLoader
train_ds = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_ds = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=True)

# Set up custom loss function and standard CrossEntropy for validation
cost_function = CustomLoss()
simple_cost_function = nn.CrossEntropyLoss().cuda()
# Gradient Descent for Vision Transformer
# Set model optimizer
gradient_descent = torch.optim.Adam(model.parameters(),
                                lr=0.0000004,
                                weight_decay=1e-7,)

fscores = []
y_true = []
y_pred = []
count = 0

# -------------------- Training loop -------------------- #
for epoch in range(50):
    losses = []
    # Loop through batches in train dataset
    for i, (batch, target) in enumerate(train_ds):
        # Increment number of steps
        count += 1
        # Empty cache for memory conservation
        torch.cuda.empty_cache()
        # Convert batch and targets to CUDA
        batch = batch.cuda()
        target = target.cuda()

        # Get model predictions
        out = model(batch)
        
        # Get loss and append it to the loss list
        cost = cost_function(out, target, count)
        losses.append(cost)

        # Update model weights and optimzer learning rate
        cost.backward()
        gradient_descent.step()
        
        # Log progress to consolve (Can remove)
        if i % 50 == 0:
            print("Epoch: {} ({}/{}) - Time: {} - Loss: {}".format(epoch, i, len(train_ds), time.time() - e_time, cost))
            e_time = time.time()


y_true_raw = []
y_pred_raw = []
all_score = []
all_targets = []
for i in range(len(test_ds.dataset.class_to_idx)):
    all_score.append([])

# -------------------- Testing loop -------------------- #
for epoch in range(15):
    # Loop through batches in test dataset
    for i, (batch, target) in enumerate(test_ds):
        # CLear cache for memory consrvation
        torch.cuda.empty_cache()
        # Load softmax function on rows
        softmax_function = torch.nn.Softmax(dim=1)
        # Set batch and target values to CUDA
        batch = batch.cuda()
        target = target.cuda()

        # Get model predictions
        out = model(batch)

        # Loop through all classes
        for i in range(len(test_ds.dataset.class_to_idx)):
            # Extend all_score list with scores for current class
            all_score[i].extend(x.item() for x in softmax_function(out.data)[:, i])
        # Convert target back to CPU and extended all_targets list
        all_targets.extend(target.cpu())
        # Computer normal loss
        cost = simple_cost_function(out, target)
         
        # Get metrics
        precision, recall, fscore, support = precision_recall_fscore_support(target.cpu(), torch.argmax(softmax_function(out.data), dim=1).cpu(),
                                                                                zero_division=0, labels=(0,1,2,3,4))

        fscores.append(fscore)
        y_true_raw.append(target.cpu())
        y_pred_raw.append(torch.argmax(softmax_function(out.data), dim=1).cpu())
        
        # Log step results to Weights and Biases
        wandb.log({"Validation loss" : cost})
        idx = 0
        for key, value in test_ds.dataset.class_to_idx.items():
            wandb.log({"Validation fscore " + key : fscore[idx]})
            idx += 1
        idx = 0
        for key, value in test_ds.dataset.class_to_idx.items():
            wandb.log({"Validation recall " + key : recall[idx]})
            idx += 1
        idx = 0
        for key, value in test_ds.dataset.class_to_idx.items():
            wandb.log({"Validation precision " + key : precision[idx]})
            idx += 1

        # Log progress to consolve 
        if i % 50 == 0:
            print("\tTest: {} ({}/{}) - Loss: {}".format(epoch, i, len(test_ds), cost))

# Turn list of lists into one long list
y_true = []
y_pred = []
for i in range(len(y_true_raw)):
    y_true.extend(y_true_raw[i])
    y_pred.extend(y_pred_raw[i])

# Get item out of tensor, list of ints and float now
for i in range(len(y_true)):
    y_true[i] = y_true[i].item()
    y_pred[i] = y_pred[i].item()

# Get list of classes
classes = [key for key, value in test_ds.dataset.class_to_idx.items()]

# Draw ROC Curves for each class
for i in range(len(classes)):
    plot_ROC_curve(all_targets, all_score[i], test_ds, i, "ROC-Curve-ViT-Inter")

# Get average F1-score for each class
out_fscores = []
for i in range(5):
    sum_class = 0
    for j in range(len(fscores)):
        sum_class += fscores[j][i]
    out_fscores.append(sum_class/len(fscores))

# Print F1-scores for each class
idx = 0
for key, value in test_ds.dataset.class_to_idx.items():
    print("Validation F1-Score " + key + ": " + str(out_fscores[idx]))
    idx += 1

# Set up confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize = (12,7))
sns.heatmap(df_cm, annot=True, fmt = "d")
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.show()