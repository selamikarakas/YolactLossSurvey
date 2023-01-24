In recent years, object detection and instance segmentation have become important tasks in computer vision, with numerous applications in robotics, surveillance, and autonomous driving. One of the state-of-the-art methods for these tasks is [YOLACT](https://github.com/dbolya/yolact), which combines the strengths of region proposal-based and anchor-based approaches. However, a critical aspect of YOLACT's performance is the choice of loss function, which determines how the model learns from training data and generalizes to unseen data. In this paper, we investigate the effect of different loss functions on the mask mAP performance of YOLACT on several benchmark datasets. 

Our changes on the source code of YOLACT is in this [file](https://github.com/selamikarakas/YolactLossSurvey/blob/main/layers/modules/multibox_loss.py). Also we added configuartions to this [file](https://github.com/selamikarakas/YolactLossSurvey/blob/main/data/config.py) to train accordingly.

In machine learning and deep learning, a loss function is a measure of how well a model is able to predict the expected output given a set of inputs. The goal of training a model is to minimize the loss function, so that the model can make accurate predictions on unseen data. There are many different types of loss functions that can be used, depending on the task at hand. In this project, we have used following loss functions:

**MSE Loss:** 

MSE (Mean Squared Error) Loss is a common loss function used in regression tasks, where the goal is to predict a continuous value. It is defined as the mean of the squared difference between the predicted value and the true value. MSE Loss is sensitive to large errors, as the square function amplifies the difference between the predicted and true values.

**Huber Loss:**

Huber Loss is a loss function that is similar to MSE Loss, but it is less sensitive to large errors. It is defined as the mean of the squared difference between the predicted and true values, but it uses a linear function instead of a square function for errors that are above a certain threshold. This makes Huber Loss more robust to outliers, as it does not amplify the effect of large errors as much as MSE Loss does.

**Dice Loss:**

Dice Loss is a loss function used in image segmentation tasks, where the goal is to predict a binary mask for each object in the image. It is defined as the negative of the Dice coefficient, which is a measure of the overlap between the predicted mask and the true mask. The Dice coefficient is calculated as the ratio of the intersection of the two masks to the union of the two masks. Dice Loss can be used to optimize the accuracy of the predicted masks.

**Log Cosh Dice Loss:** 

Log Cosh Dice Loss is a variant of Dice Loss that uses the log-cosh function instead of the square function. The log-cosh function is defined as the logarithm of the hyperbolic cosine of the difference between the predicted and true values. Log Cosh Dice Loss is less sensitive to large errors than MSE Loss and Huber Loss, making it more robust to outliers.

The results of our experiments are summarized in the following table, which shows the mask mAP scores for each loss function on Cigarette Butts Dataset.

# OUR MASK mAP and FPS RESULT ON YOLACT++ trained on [Cigarette Butts Dataset](https://www.immersivelimit.com/datasets/cigarette-butts)

|                  |  FPS  |  all  | AP50  | AP55  | AP60  | AP65  | AP70  | AP75  | AP80  | AP85  | AP90  | AP95  |
|------------------|:-----:|:-----:|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
|        BCE       | 22.21 | 83.65 | 99.99 | 99.99 | 99.99 | 99.99 | 98.99 | 98.01 | 98.01 | 89.95 | 49.81 |  1.82 |
|     Dice Loss    | 22.37 |  0.34 |  1.87 |  1.15 |  0.36 |  0.09 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |
|    Huber Loss    | 21.89 | 80.53 | 99.96 | 99.96 | 98.97 | 98.97 | 97.98 | 97.00 | 95.79 | 85.39 | 30.20 |  1.05 |
|  Log Cosh Dice   | 21.35 | 71.56 | 98.96 | 98.96 | 98.96 | 98.94 | 97.98 | 95.86 | 83.48 | 39.03 |  3.40 |  0.00 |
|     MSE Loss     | 22.89 | 79.21 | 98.95 | 98.95 | 98.95 | 97.96 | 97.96 | 96.94 | 93.60 | 82.84 | 25.92 |  0.02 |
| Huber Loss + BCE | 23.89 | 81.36 | 99.00 | 99.00 | 99.00 | 99.00 | 98.00 | 97.00 | 95.72 | 83.89 | 42.59 |  0.33 |
| Huber Loss + MSE | 21.78 | 78.39 | 99.85 | 99.85 | 99.85 | 98.87 | 98.86 | 97.80 | 93.35 | 75.13 | 20.33 | 0.00  |
|     BCE + MSE    | 21.48 | 80.54 | 99.78 | 99.78 | 99.78 | 99.78 | 98.86 | 98.79 | 96.75 | 83.25 | 28.58 |  0.05 |

Based on our experiments, we were unable to find a loss function that consistently outperformed YOLACT's original loss on all metrics. In some cases, the other loss functions performed better on certain metrics, but they did not consistently outperform the original loss across all metrics. 

# OUR MASK mAP and FPS RESULT ON YOLACT++ trained on Cigarette Butts Dataset using hybrid losses with different ratios

|                  |  α    |  β    | FPS   | all   | AP50  | AP60  | AP70  | AP80  | AP90  | AP95  
|------------------|:-----:|:-----:|-------|-------|-------|-------|-------|-------|-------|-------
|  2MSE + BCE      | 2 | 1| 27.46 | 80.53 | 99.50 | 99.50 | 98.52 | 97.42 | 30.76 | 0.13 | 
|     3MSE + BCE    | 3 |  1 |  25.28 |  79.01 |  99.02 |  99.02 |  98.07 |  93.27 |  27.95 |  0.00 |  
|    2MSE + Huber    | 2 | 1 | 26.54 | 75.44 | 99.02 | 97.13 | 97.13 | 92.18 | 14.63 | 0.00 | 
|  3MSE + Huber  | 3 | 1 | 26.40 | 78.00 | 98.99| 98.01 | 97.01 | 93.71 | 21.20| 0.00 |  
|     2Huber + BCE    | 2| 1 | 26.59 | 82.37 | 99.96 | 98.75 | 96.28 | 96.09 | 48.86| 0.82| 
|   3Huber + BCE  | 3 | 1 | 26.70 | 83.24 | 98.91 | 98.91 | 98.91 | 96.85 | 51.82| 0.38 | 
| 0.5MSE + BCE  | 0.5 | 1 | 27.46 | 84.43 | 99.98 | 99.98 | 99.98 | 97.98 | 54.41 | 2.08 | 
| 0.5MSE + Huber | 0.5 | 1 | 26.07 | 71.37 | 93.93 | 92.71 | 89.80 | 84.60 | 13.13 | 0.00 | 
|     0.5 Huber + BCE   | 0.5 | 1 | 27.62 | 82.35 | 99.86 | 99.86 | 98.89 | 97.84 | 42.13 | 83.25 | 

We later focused on hybrid losses to see if we can make further improvements in the performance of the loss functions. With the combination of 0.5MSE and BCE we were able to get better mask mAP results than the original one on Cigarette Butts Dataset.
