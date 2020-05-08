import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def focal_loss(x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,D].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        p = x.sigmoid()
        pt = p*y + (1-p)*(1-y)         # pt = p if t > 0 else 1-p
        w = alpha*y + (1-alpha)*(1-y)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, y, w, size_average=False)

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    #loss = focal_loss(logits, labels)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    #eval_score, eval_loss = evaluate(model, eval_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = train_score / len(train_loader.dataset)
        model.eval()
        eval_score, eval_loss = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval_loss: %.2f, eval score: %.2f' % (eval_loss, 100 * eval_score))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader):
    pred_result = []
    all_a = []
    logits_result = []
    with torch.no_grad():
        for v, q, a in iter(dataloader):
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()
            pred = model(v, q, None)
            x = torch.sigmoid(pred)
            pred_result.append(x.cpu().data.numpy())
            all_a.append(a)
            logits_result.append(pred)
        result = np.concatenate(pred_result, axis=0)
        result_ids = np.argmax(result, axis=1)
        all_a = torch.cat(all_a, 0)
        logits_result = torch.cat(logits_result, 0)
        one_hots = np.eye(result.shape[1])[result_ids]
        one_hots = Variable(torch.FloatTensor(one_hots)).cuda()
        score = torch.sum(torch.sum((one_hots * all_a), 1), 0)
    # loss = focal_loss(logits_result, all_a)
        loss = instance_bce_with_logits(logits_result, all_a)
    return score.cpu() / float(result_ids.shape[0]), loss
