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
import dataset
from sklearn.metrics import RocCurveDisplay

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, epoch):
        dx = -1
        dx_i = -0.65
        denom = 4 ** -3
        denom_i = 4 ** -3
        min_val = 0.01
        correction_decrease = (epoch ** dx) / denom
        correction_increase = (epoch ** dx_i) / denom_i + min_val
        criterion = nn.CrossEntropyLoss().cuda()
        init_loss = criterion(output, target)
        loss = init_loss
        for i in range(len(target)):
            if(target[i] == 2 and torch.argmax(output[i]) == 3):
                loss = init_loss + correction_increase

        if loss < 0 or (loss  / init_loss < 0.2):
            return init_loss
        return loss

def plot_ROC_curve(target, output, train_ds, positive_class, name):
    pdb.set_trace()
    positive_key = list(train_ds.dataset.class_to_idx.keys())[positive_class]
    roc_target = [int(x.item() == positive_class) for x in target]
    scores =  [x for x in output]
    fpr, tpr, thresholds = metrics.roc_curve(roc_target, scores, pos_label=1)

    RocCurveDisplay.from_predictions(
        roc_target,
        scores,
        name=f"{positive_key} vs the rest",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\n vs (Setosa & Versicolor)")
    plt.legend()
    plt.savefig(f"ROC_plots/ROC_{name}_{positive_key}.png")
    return(True)

def get_TP_TN_FP_FN(y_true, y_pred_raw):
    # Convert the raw predictions to a list of labels
    y_pred = []
    for i in range(len(y_pred_raw)):
        y_pred.append(torch.argmax(y_pred_raw[i]))
        
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    TP = TP.astype(float)
    FN = FN.astype(float)
    TN = TN.astype(float)
    
    return TP, TN, FP, FN

print(torch.__version__)
# should be 1.8.0

torch.manual_seed(2023)
wandb.init(project="ViT", name="ViT-Inter")
wandb.config = {
    "learning_rate": 0.000004,
    "epochs": 50,
    "batch_size": 8
}

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained = True)
model.head = torch.nn.Linear(in_features = 768, out_features = 5, bias = True)
model.head_dist = torch.nn.Linear(in_features = 768, out_features = 5, bias = True)
model.eval()
model = model.cuda()
pdb.set_trace()
transform = transforms.Compose([
    transforms.Resize(224, interpolation=3),
    # transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

# train_ds, _, _,test_ds = dataset.cifar10_unsupervised_dataloaders()
train_imgs = torchvision.datasets.ImageFolder(
    root = "Split_Data copy/training",
    transform=transform
)

test_imgs = torchvision.datasets.ImageFolder(
    root = "Split_Data copy/validation",
    transform=transform
)

batch_size = 32
train_ds = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_ds = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=True)

cost_function = CustomLoss()
simple_cost_function = nn.CrossEntropyLoss().cuda()
# Gradient Descent for Vision Transformer
gradient_descent = torch.optim.Adam(model.parameters(),
                                lr=0.0000004,
                                weight_decay=1e-7,)

fscores = []
y_true = []
y_pred = []
count = 0
for epoch in range(50):
    losses = []
    e_time = time.time()
    for i, (batch, target) in enumerate(train_ds):
        count += 1
        torch.cuda.empty_cache()
        batch = batch.cuda()
        target = target.cuda()
        out = model(batch)
        
        cost = cost_function(out, target, count)
        losses.append(cost)
        cost.backward()
        gradient_descent.step()
        
        if i % 50 == 0:
            print("Epoch: {} ({}/{}) - Time: {} - Loss: {}".format(epoch, i, len(train_ds), time.time() - e_time, cost))
            e_time = time.time()

y_true_raw = []
y_pred_raw = []
all_score = []
all_targets = []

for i in range(len(test_ds.dataset.class_to_idx)):
    all_score.append([])
for epoch in range(15):
    for i, (batch, target) in enumerate(test_ds):
        torch.cuda.empty_cache()
        softmax_function = torch.nn.Softmax(dim=1) 
        batch = batch.cuda()
        target = target.cuda()
        out = model(batch)
        for i in range(len(test_ds.dataset.class_to_idx)):
            all_score[i].extend(x.item() for x in softmax_function(out.data)[:, i])
        all_targets.extend(target.cpu())
        cost = simple_cost_function(out, target)
         
        precision, recall, fscore, support = precision_recall_fscore_support(target.cpu(), torch.argmax(softmax_function(out.data), dim=1).cpu(),
                                                                                zero_division=0, labels=(0,1,2,3,4))

        fscores.append(fscore)
        full_e_time = time.time()
        y_true_raw.append(target.cpu())
        y_pred_raw.append(torch.argmax(softmax_function(out.data), dim=1).cpu())
        
        # No point of following two lines of code
        y_true.extend(target.cpu())
        y_pred.extend(torch.argmax(softmax_function(out.data), dim=1))
        
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
            
        if i % 50 == 0:
            print("\tTest: {} ({}/{}) - Loss: {}".format(epoch, i, len(test_ds), cost))

y_true = []
y_pred = []
for i in range(len(y_true_raw)):
    y_true.extend(y_true_raw[i])
    y_pred.extend(y_pred_raw[i])

for i in range(len(y_true)):
    y_true[i] = y_true[i].item()
    y_pred[i] = y_pred[i].item()
    
fpr = dict()
tpr = dict()
roc_auc = dict()
classes = [key for key, value in test_ds.dataset.class_to_idx.items()]

for i in range(len(classes)):
    plot_ROC_curve(all_targets, all_score[i], test_ds, i, "ROC-Curve-ViT-Inter")

out_fscores = []
for i in range(5):
    sum_class = 0
    for j in range(len(fscores)):
        sum_class += fscores[j][i]
    out_fscores.append(sum_class/len(fscores))

idx = 0
for key, value in test_ds.dataset.class_to_idx.items():
    print("Validation F1-Score " + key + ": " + str(out_fscores[idx]))
    idx += 1

cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                     columns = [i for i in classes])

fig, ax = plt.subplots(figsize = (12,7))
sns.heatmap(df_cm, annot=True)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.show()
































# model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
# model.eval()
# scripted_model = torch.jit.script(model)
# scripted_model.save("fbdeit_scripted.pt")

# # Use 'fbgemm' for server inference and 'qnnpack' for mobile inference
# backend = "fbgemm" # replaced with qnnpack causing much worse inference speed for quantized model on this notebook
# model.qconfig = torch.quantization.get_default_qconfig(backend)
# torch.backends.quantized.engine = backend

# quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
# scripted_quantized_model = torch.jit.script(quantized_model)
# scripted_quantized_model.save("fbdeit_scripted_quantized.pt")

# out = scripted_quantized_model(img)
# clsidx = torch.argmax(out)
# print(clsidx.item())
# # The same output 269 should be printed

# from torch.utils.mobile_optimizer import optimize_for_mobile
# optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
# optimized_scripted_quantized_model.save("fbdeit_optimized_scripted_quantized.pt")

# out = optimized_scripted_quantized_model(img)
# clsidx = torch.argmax(out)
# print(clsidx.item())
# # Again, the same output 269 should be printed

# optimized_scripted_quantized_model._save_for_lite_interpreter("fbdeit_optimized_scripted_quantized_lite.ptl")
# ptl = torch.jit.load("fbdeit_optimized_scripted_quantized_lite.ptl")

# # import pdb; pdb.set_trace()
# with torch.autograd.profiler.profile(use_cuda=True) as prof1:
#     out = model(img)
# with torch.autograd.profiler.profile(use_cuda=True) as prof2:
#     out = scripted_model(img)
# with torch.autograd.profiler.profile(use_cuda=True) as prof3:
#     out = scripted_quantized_model(img)
# with torch.autograd.profiler.profile(use_cuda=True) as prof4:
#     out = optimized_scripted_quantized_model(img)
# with torch.autograd.profiler.profile(use_cuda=True) as prof5:
#     out = ptl(img)

# print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
# print("scripted model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))
# print("scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))
# print("scripted & quantized & optimized model: {:.2f}ms".format(prof4.self_cpu_time_total/1000))
# print("lite model: {:.2f}ms".format(prof5.self_cpu_time_total/1000))

# import pandas as pd
# import numpy as np

# df = pd.DataFrame({'Model': ['original model','scripted model', 'scripted & quantized model', 'scripted & quantized & optimized model', 'lite model']})
# df = pd.concat([df, pd.DataFrame([
#     ["{:.2f}ms".format(prof1.self_cpu_time_total/1000), "0%"],
#     ["{:.2f}ms".format(prof2.self_cpu_time_total/1000),
#      "{:.2f}%".format((prof1.self_cpu_time_total-prof2.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
#     ["{:.2f}ms".format(prof3.self_cpu_time_total/1000),
#      "{:.2f}%".format((prof1.self_cpu_time_total-prof3.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
#     ["{:.2f}ms".format(prof4.self_cpu_time_total/1000),
#      "{:.2f}%".format((prof1.self_cpu_time_total-prof4.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
#     ["{:.2f}ms".format(prof5.self_cpu_time_total/1000),
#      "{:.2f}%".format((prof1.self_cpu_time_total-prof5.self_cpu_time_total)/prof1.self_cpu_time_total*100)]],
#     columns=['Inference Time', 'Reduction'])], axis=1)

# print(df)