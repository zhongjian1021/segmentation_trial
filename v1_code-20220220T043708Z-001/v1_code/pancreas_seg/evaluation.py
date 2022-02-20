# 计算一些评价指标的



# 分割常用的评价指标
import numpy as np
import torch



def get_accuracy(SR,GT,threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = SR > threshold
    GT = GT == np.max(GT)
    corr = np.sum(SR==GT)

    acc = float(corr)/float(SR.shape[0])

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float)
    GT = (GT == np.max(GT)).astype(np.float)

    # TP : True Positive
    # FN : False Negative
    TP = (((SR == 1.).astype(np.float) + (GT == 1.).astype(np.float)) == 2.).astype(np.float)
    FN = (((SR == 0.).astype(np.float) + (GT == 1.).astype(np.float)) == 2.).astype(np.float)

    SE = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)

    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR.view(-1)
    GT = GT.view(-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()

    SR = (SR > threshold).astype(np.float)
    GT = (GT == np.max(GT)).astype(np.float)

    # TN : True Negative
    # FP : False Positive
    TN = (((SR == 0.).astype(np.float) + (GT == 0.).astype(np.float)) == 2.).astype(np.float)
    FP = (((SR == 1.).astype(np.float) + (GT == 0.).astype(np.float)) == 2.).astype(np.float)

    SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)

    return SP

def get_DC(SR,GT,threshold=0.5, reduce = True):
    # DC : Dice Coefficient
    SR = SR.view(SR.shape[0],-1)
    GT = GT.view(GT.shape[0],-1)

    SR = SR.data.cpu().numpy()
    GT = GT.data.cpu().numpy()


    SR = (SR > threshold).astype(np.float)
    GT = (GT == np.max(GT)).astype(np.float)

    Inter = ((SR+GT)==2).astype(np.float)

    DC_case_wised = 2*Inter.sum(1) /((SR.sum(1)+GT.sum(1)) + 1e-6)

    if reduce:
        return DC_case_wised.mean()
    else:
        return list(DC_case_wised)
