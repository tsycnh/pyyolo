import cv2

import csv
import sys,os

import calc_iou

# 可视化通过 ./darknet detector valid 生成的文件
# 参数分别为 文件路径，原始图像目录，扩展名
# 如：../results/comp4_det_test_lines.txt /Users/shidanlifuhetian/All/data/lines_train_darknet jpg 0.5

# 该文件仅仅可视化一些未标记的图像分割结果。每张分割图都会被保存下来

def format_voc_label(label):

    x = int(float(label[2]))
    y = int(float(label[3]))
    x2 = int(float(label[4]))
    y2 = int(float(label[5]))

    return [x,y,x2,y2]

def format_darknet_label(label,img_width,img_height):

    center_x = int(float(label[1]) * img_width)
    center_y = int(float(label[2]) * img_height)
    width = int(float(label[3]) * img_width)
    height = int(float(label[4]) * img_height)

    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    return [x1,y1,x2,y2]

# 绘制voc格式的bbox
def draw_info(img,rows,color=(0,0,255),threshold=0.5):
    bbox_darwed = 0
    for row in rows:
        confidence = float(row[1])
        if confidence < threshold:
            break
        x = int(float(row[2]))
        y = int(float(row[3]))
        x2 = int(float(row[4]))
        y2 = int(float(row[5]))

        cv2.rectangle(img, (x, y), (x2, y2),color=color)
        cv2.putText(img, str(round(confidence, 2)), (x2, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, color)
        bbox_darwed+=1
    return img,bbox_darwed

# 绘制darknet格式的bbox
def draw_gt(img,rows,color=(0,255,0)):

    bbb = img.shape
    img_height = bbb[0]
    img_width = bbb[1]

    for row in rows:
        [x1,y1,x2,y2] = format_darknet_label(row,img_width,img_height)

        cv2.rectangle(img, (x1,y1), (x2, y2), color)
        cv2.putText(img, 'gt', (x1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 1, color)
    return img

def calc_score(predicts,gts):
    # 分别计算recall 和precision
    for predict in predicts:
        if float(predict[1]) < threshold:
            predicts.remove(predict)


    score={
        'recall_base':len(gts),
        'precision_base':len(predicts),
        'match':0
    }

    for gt in gts:
        bb_gt = format_darknet_label(gt,128,128)
        for predict in predicts:
            bb_pd = format_voc_label(predict)
            iou = calc_iou.iou(bb_gt,bb_pd)
            if iou > 0.5:
                score['match'] += 1

    return score

if __name__ == "__main__":
    csv_file = sys.argv[1]
    data_folder = sys.argv[2]
    extension = "."+sys.argv[3]
    threshold = float(sys.argv[4])

    f = open(csv_file, 'r')
    reader = csv.reader(f,delimiter=' ')

    row_buffer = []
    tmp_folder = './tmp'
    marked_folder = tmp_folder+'/marked/'
    unknown_folder = tmp_folder+'/unknown/'
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    if not os.path.exists(marked_folder):
        os.makedirs(marked_folder)
    if not os.path.exists(unknown_folder):
        os.makedirs(unknown_folder)
    for row in reader:
        print(row)

        if len(row_buffer) > 0:
            if row_buffer[0][0] == row[0]:
                row_buffer.append(row)
            else:
                file_name = row_buffer[0][0].replace('\\', '/')
                image_full_name = data_folder + "/JPEGImages/" + file_name + extension
                # /Users/shidanlifuhetian/All/data/lines_test_darknet/JPEGImages
                img = cv2.imread(image_full_name)

                img,bbox_drawed = draw_info(img,row_buffer,threshold=threshold)

                # cv2.imshow(image_full_name, img)
                # cv2.waitKey(1000)
                if bbox_drawed>0:
                    cv2.imwrite(marked_folder+file_name+extension,img)
                else:
                    cv2.imwrite(unknown_folder+file_name+extension,img)

                row_buffer = []  # clear buffer
                row_buffer.append(row)
        else:# empty
            row_buffer.append(row)

    f.close()
