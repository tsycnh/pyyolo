some assistant scripts for darknet yolo

visualize_result.py

visualize the image of ground truth bboxes and predicted bboxed. Also calculating
the precision and recall.

usage

visualize_result.py \[/path/to/valid/result/file\] \[path/to/images/folder\] \[image_extension\] \[threshold\]

eg:
visualize_result.py /Users/shidanlifuhetian/All/Tdevelop/darknet/results/comp4_det_test_lines.txt /Users/shidanlifuhetian/All/data/lines_train_darknet jpg 0.5
