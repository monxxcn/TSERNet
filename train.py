import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import glob
from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from model import TSERNet
import pytorch_ssim
import pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def floss(prediction, target, beta=0.3, log_like=False):
    EPS = 1e-10
    N = N = prediction.size(0)
    TP = (prediction * target).view(N, -1).sum(dim=1)
    H = beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
    fmeasure = (1 + beta) * TP / (H + EPS)
    if log_like:
        floss = -torch.log(fmeasure)
    else:
        floss = (1 - fmeasure)

    return floss.mean()


def get_multi_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)
    floss_out = floss(pred, target)

    loss = bce_out + ssim_out + iou_out + floss_out

    return loss


def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy(input, target, weights, reduction=reduction)


def get_total_loss(d0, d1, d2, d3, d4, d5, d6, db, d_edge, d1_ref, d2_ref, d3_ref, d4_ref, labels_v, sup_v):
    loss0 = get_multi_loss(d0, labels_v)
    loss1 = get_multi_loss(d1, labels_v)
    loss2 = get_multi_loss(d2, labels_v)
    loss3 = get_multi_loss(d3, labels_v)
    loss4 = get_multi_loss(d4, labels_v)
    loss5 = get_multi_loss(d5, labels_v)
    loss6 = get_multi_loss(d6, labels_v)
    lossb = get_multi_loss(db, labels_v)
    loss_edge = bce2d_new(d_edge, sup_v, reduction='elementwise_mean')
    loss_ref1 = get_multi_loss(d1_ref, labels_v)
    loss_ref2 = get_multi_loss(d2_ref, labels_v)
    loss_ref3 = get_multi_loss(d3_ref, labels_v)
    loss_ref4 = get_multi_loss(d4_ref, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + lossb + loss_edge + \
           loss_ref1 + loss_ref2 + loss_ref3 + loss_ref4
    print("RGB Loss: l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f, lb: %3f, le: %3f" % (
        loss0.item(), loss1.item(), loss2.item(), loss3.item(),
        loss4.item(), loss5.item(), loss6.item(), lossb.item(), loss_edge.item()))

    print("Ref Loss: l1: %3f, l2: %3f, l3: %3f, l4: %3f" % (
        loss_ref1.item(), loss_ref2.item(), loss_ref3.item(), loss_ref4.item()))

    return loss


# ------- 2. set the directory of training dataset --------

model = TSERNet(3, 1)

data_dir = ''

tra_image_dir = ''
tra_label_dir = ''
tra_supervision_dir = ''

image_ext = ''
label_ext = ''

model_dir = ""

epoch_num = 100000
batch_size_train = 4
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_supervision_name_list = glob.glob(data_dir + tra_supervision_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    imidx = imidx.split("\\")[-1]
    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("supervision labels: ", len(tra_supervision_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    supervision_name_list=tra_supervision_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 3. define model --------
net = TSERNet(3, 1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
ite_num4val = 0

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels, supervision = data['image'], data['label'], data['supervision']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        supervision = supervision.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v, supervision_v = Variable(inputs.cuda(), requires_grad=False), \
                                                Variable(labels.cuda(), requires_grad=False), \
                                                Variable(supervision.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v, supervision_v = Variable(inputs, requires_grad=False), \
                                                Variable(labels, requires_grad=False), \
                                                Variable(supervision, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6, db, d_edge, d1_ref, d2_ref, d3_ref, d4_ref = net(inputs_v)
        loss = get_total_loss(d0, d1, d2, d3, d4, d5, d6, db, d_edge,
                              d1_ref, d2_ref, d3_ref, d4_ref, labels_v, supervision_v)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, db, loss, d_edge, d1_ref, d2_ref, d3_ref, d4_ref

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f \n" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val))

        if ite_num % 1000 == 0:  # save model every 1000 iterations
            torch.save(net.state_dict(), model_dir + "ite_%d.pth" % (ite_num))
            running_loss = 0.0
            running_tar_loss = 0.0
            # net.train()  # resume train
            ite_num4val = 0

print('-------------Congratulations! Training Done!!!-------------')
