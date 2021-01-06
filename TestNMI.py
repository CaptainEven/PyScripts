# encoding=utf-8
import math
import numpy as np
from sklearn import metrics


def NMI(A, B):
    # 样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)

    # 互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A == idA)
            idBOccur = np.where(B == idB)

            idABOccur = np.intersect1d(idAOccur, idBOccur)

            px = 1.0*len(idAOccur[0]) / total
            py = 1.0*len(idBOccur[0]) / total
            pxy = 1.0*len(idABOccur) / total
            MI = MI + pxy*math.log(pxy/(px*py)+eps, 2)
    print("MI: %.3f" % (MI))

    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps, 2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B == idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps, 2)

    print("HA: %.3f" % (Hx))
    print("HB: %.3f" % (Hy))
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat


def Test2():
    """
    B=A or
    B=T(A): There is a 1-1 transformation between A and B
    """
    A = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    B = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    P11 = 6 / 17
    P12 = 0.0
    P13 = 0.0

    P21 = 0.0
    P22 = 6 / 17
    P23 = 0.0

    P31 = 0.0
    P32 = 0.0
    P33 = 5 / 17

    P1 = 6 / 17
    P2 = 6 / 17
    P3 = 5 / 17

    P1_ = 6 / 17
    P2_ = 6 / 17
    P3_ = 5 / 17

    eps = 1.4e-45
    MI = 0.0
    MI += P11 * math.log2(P11 / (P1 * P1) + eps)
    MI += P12 * math.log2(P12 / (P1 * P2) + eps)
    MI += P13 * math.log2(P13 / (P1 * P3) + eps)

    MI += P21 * math.log2(P21 / (P2 * P1) + eps)
    MI += P22 * math.log2(P22 / (P2 * P2) + eps)
    MI += P23 * math.log2(P23 / (P2 * P3) + eps)

    MI += P31 * math.log2(P31 / (P3 * P1) + eps)
    MI += P32 * math.log2(P32 / (P3 * P2) + eps)
    MI += P33 * math.log2(P33 / (P3 * P3) + eps)

    # MI = P11 * math.log2(P11 / (P1 * P1)) + \
    #     P12 * math.log2(P12 / (P1 * P2)) + \
    #     P13 * math.log2(P13 / (P1 * P3)) + \
    #     P21 * math.log2(P21 / (P2 * P1)) + \
    #     P22 * math.log2(P22 / (P2 * P2)) + \
    #     P23 * math.log2(P23 / (P2 * P3)) + \
    #     P31 * math.log2(P31 / (P3 * P1)) + \
    #     P32 * math.log2(P32 / (P3 * P2)) + \
    #     P33 * math.log2(P33 / (P3 * P3))
    print("MI: %.3f" % (MI))

    HA = -1.0 * P1 * math.log2(P1) - 1.0 * P2 * math.log2(P2) \
         - 1.0 * P3 * math.log2(P3)
    HB = -1.0 * P1_ * math.log2(P1_) - 1.0 * P2_ * math.log2(P2_) \
         - 1.0 * P3_ * math.log2(P3_s)
    print("H(A): %.3f" % (HA))
    print("H(B): %.3f" % (HB))

    NMI = 2.0 * MI / (HA + HB)

    print("NMI(A, B): %.3f" % (NMI))


def Test1():
    """
    """
    A = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    # B = 2*np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    # B = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    B = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])

    print("A:\n", A)
    print("B:\n", B)
    print("NMI(A, B):\n", NMI(A, B))
    # print(metrics.normalized_mutual_info_score(A, B))

if __name__ == '__main__':
    Test1()
    # Test2()
