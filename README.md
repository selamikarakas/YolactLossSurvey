In recent years, object detection and instance segmentation have become important tasks in computer vision, with numerous applications in robotics, surveillance, and autonomous driving. One of the state-of-the-art methods for these tasks is [YOLACT](https://github.com/dbolya/yolact), which combines the strengths of region proposal-based and anchor-based approaches. However, a critical aspect of YOLACT's performance is the choice of loss function, which determines how the model learns from training data and generalizes to unseen data. In this paper, we investigate the effect of different loss functions on the mask mAP performance of YOLACT on several benchmark datasets. 

Our changes on the source code of YOLACT is in this [file](https://github.com/selamikarakas/YolactLossSurvey/blob/main/layers/modules/multibox_loss.py). Also we added configuartions to this [file](https://github.com/selamikarakas/YolactLossSurvey/blob/main/data/config.py) to train accordingly.

# OUR MASK mAP and FPS RESULT ON YOLACT++ trained on [Cigarette Butts Dataset](https://www.immersivelimit.com/datasets/cigarette-butts)

|                  |  FPS  |  all  | AP50  | AP55  | AP60  | AP65  | AP70  | AP75  | AP80  | AP85  | AP90  | AP95  |
|------------------|:-----:|:-----:|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|        BCE       | 22.21 | 83.65 | 99.99 | 99.99 | 99.99 | 99.99 | 98.99 | 98.01 | 98.01 | 89.95 | 49.81 |  1.82 |
|     Dice Loss    | 22.37 |  0.34 |  1.87 |  1.15 |  0.36 |  0.09 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |
|    Huber Loss    | 21.89 | 80.53 | 99.96 | 99.96 | 98.97 | 98.97 | 97.98 | 97.00 | 95.79 | 85.39 | 30.20 |  1.05 |
|  Log Cosh Dice   | 21.35 | 71.56 | 98.96 | 98.96 | 98.96 | 98.94 | 97.98 | 95.86 | 83.48 | 39.03 |  3.40 |  0.00 |
|     MSE Loss     | 22.89 | 79.21 | 98.95 | 98.95 | 98.95 | 97.96 | 97.96 | 96.94 | 93.60 | 82.84 | 25.92 |  0.02 |
|   Log Cosh BCE   | 21.46 | 80.65 | 99.78 | 99.78 | 99.78 | 98.84 | 98.84 | 97.90 | 95.47 | 82.11 | 32.16 |  1.84 |
| Huber Loss + BCE | 23.89 | 81.36 | 99.00 | 99.00 | 99.00 | 99.00 | 98.00 | 97.00 | 95.72 | 83.89 | 42.59 |  0.33 |
| Huber Loss + MSE | 21.78 | 78.39 | 99.85 | 99.85 | 99.85 | 98.87 | 98.86 | 97.80 | 93.35 | 75.13 | 20.33 | 0.00  |
|     BCE + MSE    | 21.48 | 80.54 | 99.78 | 99.78 | 99.78 | 99.78 | 98.86 | 98.79 | 96.75 | 83.25 | 28.58 |  0.05 |
