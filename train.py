from dassl.metrics import compute_accuracy
import copy
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from dspl_model import LayerNorm
from tensorboardX import SummaryWriter
from utils import compute_oscr

writer = SummaryWriter('tens/')

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# model_tmp, preprocess = clip.load('ViT-B/32', 'cuda:0')

def compute_logits(image_features, text_features):
    ln_post = LayerNorm(512)
    image_features = ln_post(image_features[:, 0, :])

    b, _ = image_features.size()
    text_features = text_features[:20, :]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.t()

    return logits


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


focal_loss = FocalLoss(20)


def draw_picture(act_close, act_open):
    plt.style.use('classic')
    n, bins, patches = plt.hist(act_close, 50, facecolor='green', alpha=0.5, histtype='bar', label='unknown',
                                edgecolor='white')
    n, bins, patches = plt.hist(act_open, 50, facecolor='blue', alpha=0.5, histtype='bar', label='unknown',
                                edgecolor='white')
    plt.grid(False)
    plt.close()


def class_dis(target_close, pred_close, pred_open, act_close, act_open, class_want):
    pred_close = torch.tensor([item.cpu().detach().numpy() for item in pred_close])
    pred_open = torch.tensor([item.cpu().detach().numpy() for item in pred_open])

    act_close_pred_tmp = np.reshape(np.array(pred_close), (-1))
    act_open_pred = np.reshape(np.array(pred_open), (-1))

    label_want_idx_close = []
    label_want_idx_open = []

    for i, j in enumerate(act_open_pred):
        if j == class_want:
            label_want_idx_open.append(i)

    act_open = act_open[label_want_idx_open]

    print(act_open.shape)

    idx_open = []
    for i, j in enumerate(act_open):
        if j >= 15:
            idx_open.append(i)
    for i in idx_open:
        print(label_want_idx_open[i], act_open_pred[label_want_idx_open[i]])


def test(model, testloader, args):
    model.eval()
    output_list = []
    out_label = []
    loss_list = []
    t = 0
    cls_num = len(args.train_classes)

    with torch.no_grad():
        for batch_idx, (image, label, idx) in enumerate(tqdm(testloader)):
            t = batch_idx
            image, label = image.to(device), label.to(device)
            out_label.append(label)
            output, _, _, _, _ = model(image)

            output_list.append(output[:, 0:cls_num])
            out = torch.cat(output_list)

    labels = torch.cat(out_label)
    acc = compute_accuracy(out, labels)[0].item()
    print('acc:', acc)
    return acc


def test_openset(model, testloader, outloader, args):
    model.eval()
    act_close = []
    # act_close_fake = []
    target_close = []
    pred_close = []
    logits_per_image_list = []
    cls_num = len(args.train_classes)
    with torch.no_grad():
        for batch_idx, (images, labels, idx) in enumerate(tqdm(testloader)):
            images, labels = images.to(device), labels.to(device)
            target_close.append(labels)
            logits_per_image, _, _, _, _ = model(images)
            # logits_per_image_un = logits_per_image[:, cls_num * 2]
            logits_per_image_fake = logits_per_image[:, cls_num:cls_num * 2]
            logits_per_image = logits_per_image[:, 0:cls_num]
            logits_per_image_list.append(logits_per_image)
            values, indices = torch.max(logits_per_image, dim=1)
            pred_close.append(indices)
            prob = F.softmax(logits_per_image, dim=1)
            for i in range(prob.size(0)):
                values, indices_pre = torch.topk(logits_per_image[i], 1, dim=0, largest=True, sorted=True, out=None)
                values_fake, _ = torch.topk(logits_per_image_fake[i], 1, dim=0, largest=True, sorted=True, out=None)
                act = values[0].unsqueeze(0)
                ack_fake = values_fake[0].unsqueeze(0)
                act = (1 - args.alpha1) * act - args.alpha1 * ack_fake  # 减去un
                act_close.append(act)
                # act_close_fake.append(ack_fake)

        outs = torch.cat(logits_per_image_list)
        target_closes = torch.cat(target_close)
        acc = compute_accuracy(outs, target_closes)[0].item()
        print('acc:', acc)
        act_close = np.reshape(np.array(torch.cat(act_close).cpu()), (-1))
        # act_close_fake = np.reshape(np.array(torch.cat(act_close_fake).cpu()), (-1))
        # act_close = (act_close-np.min(act_close)) / (np.max(act_close) - np.min(act_close))
        # # act_close_fake = -act_close_fake
        # act_close_fake = (act_close_fake-np.min(act_close_fake)) / (np.max(act_close_fake) - np.min(act_close_fake))
        # # act_close *= act_close_fake
        # act_close = (1 - args.alpha1) * act_close - args.alpha1 * act_close_fake

        target_close = np.reshape(np.array(torch.cat(target_close).cpu()), (-1))
        pred_close = np.reshape(np.array(torch.cat(pred_close).cpu()), (-1))

    act_open = []
    # act_open_fake = []
    target_open = []
    pred_open = []
    with torch.no_grad():
        for batch_idx, (images, labels, idx) in enumerate(tqdm(outloader)):
            images, labels = images.to(device), labels.to(device)
            target_open.append(labels)
            logits_per_image, _, _, _, _ = model(images)
            # logits_per_image_un = logits_per_image[:, cls_num * 2]
            logits_per_image_fake = logits_per_image[:, cls_num:cls_num * 2]
            logits_per_image = logits_per_image[:, 0:cls_num]
            values, indices = torch.max(logits_per_image, dim=1)
            pred_open.append(indices)
            prob = F.softmax(logits_per_image, dim=1)
            for i in range(logits_per_image.size(0)):
                values, indices_pre = torch.topk(logits_per_image[i], 1, dim=0, largest=True, sorted=True, out=None)
                values_fake, _ = torch.topk(logits_per_image_fake[i], 1, dim=0, largest=True, sorted=True, out=None)
                act = values[0].unsqueeze(0)
                ack_fake = values_fake[0].unsqueeze(0)
                act = (1 - args.alpha1) * act - args.alpha1 * ack_fake  # 减去un
                act_open.append(act)
        #         act_open_fake.append(ack_fake)

        act_open = np.reshape(np.array(torch.cat(act_open).cpu()), (-1))
        # act_open_fake = np.reshape(np.array(torch.cat(act_open_fake).cpu()), (-1))
        # act_open = (act_open-np.min(act_open)) / (np.max(act_open) - np.min(act_open))
        # # act_open_fake = -act_open_fake
        # act_open_fake = (act_open_fake-np.min(act_open_fake)) / (np.max(act_open_fake) - np.min(act_open_fake))
        # # act_open *= act_open_fake
        # act_open = (1 - args.alpha1) * act_open - args.alpha1 * act_open

        target_open = np.reshape(np.array(torch.cat(target_open).cpu()), (-1))
        target_open[:] = -1

    # pred_open = pred_open[:-1]
    pred_open = np.reshape(np.array(torch.cat(pred_open).cpu()), (-1))

    act_close_solo = copy.deepcopy(act_close)
    act_open_solo = copy.deepcopy(act_open)
    auc_pred_solo = np.hstack([act_close_solo, act_open_solo])

    auc_known_labels = copy.deepcopy(target_close)
    auc_unknown_labels = copy.deepcopy(target_open)
    auc_known_labels[:] = 1
    auc_unknown_labels[:] = 0
    auc_labels = np.hstack([auc_known_labels, auc_unknown_labels])
    auc = roc_auc_score(auc_labels, auc_pred_solo)
    print('auc', auc)
    oscr = compute_oscr(act_close_solo, act_open_solo, pred_close, target_close)
    print('oscr', oscr)
    fpr, tpr, thresholds = roc_curve(auc_labels, auc_pred_solo)
    thresh = thresholds[np.abs(np.array(tpr) - 0.95).argmin()]
    predicts = np.hstack([pred_close, pred_open])
    predicts[auc_pred_solo <= thresh] = -1
    f1_known_labels = copy.deepcopy(target_close)
    f1_unknown_labels = copy.deepcopy(target_open)
    # f1_unknown_labels[:] = -1
    f1_labels = np.hstack([f1_known_labels, f1_unknown_labels])
    macrof1 = f1_score(f1_labels, predicts, average='macro')
    print('macro_f1', macrof1)

    return auc, oscr, macrof1, acc


def train_double(model_p, optimizer, dataloader, epoch, args):
    output_list = []
    out_label = []
    loss_list = []
    loss_sim = []
    loss_pos_list = []
    loss_neg_list = []
    t = 0
    cls_num = len(args.train_classes)
    model_p.train()
    # # ====== 计时开始 ======
    # import time
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # epoch_start = time.perf_counter()

    for batch_idx, (image, label, idx) in enumerate(tqdm(dataloader)):
        t = batch_idx
        image, label = image.to(device), label.to(device)
        out_label.append(label)
        output_for, output_back, output_com, out_fake, sim_fb = model_p(image, label)
        output_list.append(output_for[:, 0:cls_num])

        # for not prompt loss

        target_en = torch.Tensor(label.shape[0], len(args.target_classes)).to(device)
        target_en.zero_()
        target_en.scatter_(1, label.view(-1, 1), 1)
        soft_out = F.softmax(output_back, dim=1)
        log_soft_out = torch.log(soft_out)

        loss_neg = 0.01 * (-1 * log_soft_out).sum() / (
                 output_back.size(0) - 1)  # - F.nll_loss(exp_soft_out, label)
        # un_label = (torch.ones(label.size()[0], dtype=torch.int64) * (cls_num * 2)).to(device)
        loss_pos = F.cross_entropy(output_for, label)
        sim_fb_loss = sim_fb.sum() / sim_fb.size(0)
        sim_fb_loss = 1 / (-1 * sim_fb_loss)

        loss_fake = F.cross_entropy(out_fake, label + cls_num)
        loss = loss_pos + args.lambda1 * loss_fake + args.lambda2 * sim_fb_loss + args.lambda3 * loss_neg

        # loss = loss_pos + args.lambda1 * sim_fb_loss
        loss_list.append(loss.item())
        loss_pos_list.append(loss_pos.item())
        loss_neg_list.append(loss_neg.item())
        loss_sim.append(sim_fb_loss.item())

        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #  # ====== 计时结束 ======
    # if torch.cuda.is_available():
    #     torch.cuda.synchronize()
    # epoch_end = time.perf_counter()
    # epoch_secs = epoch_end - epoch_start
    # print(f"time: {epoch_secs:.2f} s")
    # if torch.cuda.is_available():
    #     allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
    #     max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # MB
    #     cached = torch.cuda.memory_reserved(device) / 1024 ** 2  # MB
    #     print(f"GPU Memory Allocated: {allocated:.2f} MB")
    #     print(f"GPU Max Memory Allocated: {max_allocated:.2f} MB")
    #     print(f"GPU Memory Reserved by Caching Allocator: {cached:.2f} MB")

    out = torch.cat(output_list)
    labels = torch.cat(out_label)
    loss = sum(loss_list) / (t + 1)
    loss_pos = sum(loss_pos_list) / (t + 1)
    loss_neg = sum(loss_neg_list) / (t + 1)
    loss_sim = sum(loss_sim) / (t + 1)
    writer.add_scalar('loss_pos', loss_pos, epoch)
    writer.add_scalar('loss_neg', loss_neg, epoch)
    writer.add_scalar('loss_sim', loss_sim, epoch)
    writer.add_scalar('loss_total', loss, epoch)
    acc = compute_accuracy(out, labels)[0].item()
    print('acc:', acc)
    print('loss', loss)
    return loss


