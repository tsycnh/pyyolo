import numpy as np

# 输入为两个数组，分别为四个坐标,依次是矩形左上点的x,y，右下点的x,y
def intersect(bb1,bb2):

    x1 = bb1[0]
    y1 = bb1[1]
    x2 = bb1[2]
    y2 = bb1[3]
    x3 = bb2[0]
    y3 = bb2[1]
    x4 = bb2[2]
    y4 = bb2[3]

    x_l_max = max(x1,x3)
    x_r_min = min(x2,x4)
    y_t_max = max(y1,y3)
    y_b_min = min(y2,y4)

    area = 0
    if x_l_max <= x_r_min and y_t_max<= y_b_min:
        #相交
        area = (x_r_min-x_l_max)*(y_b_min-y_t_max)

    return area

def union_set(bb1,bb2):

    area_bb1 = (bb1[2] - bb1[0])*(bb1[3] - bb1[1])
    area_bb2 = (bb2[2] - bb2[0])*(bb2[3] - bb2[1])
    area_intersect = intersect(bb1,bb2)

    return area_bb1+area_bb2-area_intersect

def iou(bb1,bb2):

    return intersect(bb1,bb2)/union_set(bb1,bb2)