# VGG Model Distillation
BUAA Graduation Project
## Frame
![frame](./img/frame-4.png)
## Experiment Record
- Analyze upsampling alignment's effect to teacher model
    - 100 epoches, batch = 32, old transform, $\lambda_{ce} = 0.1$ 

      | mode | without upsampling | upsampling |
      | ---- | ---- | ---- |
      | accuracy | 86% |  86%  |

    - 20 epoches, batch = 32, new transform, $\lambda_{ce} = 0.1, \lambda_{up} = 5$

      | mode | without upsampling | upsampling |
      | ---- | ---- | ---- |
      | accuracy | 85% |  71%  |

- Pretrain student model
    - 20 epoches, batch = 32, old transform, $\lambda_{ce} = 0.1$ 

      | mode | without upsampling | upsampling |
      | ---- | ---- | ---- |
      | accuracy | 59% |  53%  |

- Fisrt Distill Experiment
    - 20 epoches, batch = 32, $\lambda_{ce} = 0.1$, without pixel-wise distillation, accuracy: 71\%
    - 20 epoches, batch = 32, $\lambda_{ce} = 0.1$, use pixel-wise distillation, accuracy:

Accuracy of plane : 86.7257 %
Accuracy of   car : 90.4348 %
Accuracy of  bird : 80.9211 %
Accuracy of   cat : 62.0155 %
Accuracy of  deer : 87.1212 %
Accuracy of   dog : 84.9558 %
Accuracy of  frog : 90.9091 %
Accuracy of horse : 92.9688 %
Accuracy of  ship : 90.2655 %
Accuracy of truck : 97.0588 %
Average Accuracy: 86.3376 %

Accuracy of plane : 90.2655 %
Accuracy of   car : 93.0435 %
Accuracy of  bird : 82.8947 %
Accuracy of   cat : 65.1163 %
Accuracy of  deer : 81.8182 %
Accuracy of   dog : 83.1858 %
Accuracy of  frog : 92.5620 %
Accuracy of horse : 88.2812 %
Accuracy of  ship : 92.9204 %
Accuracy of truck : 93.3824 %
Average Accuracy: 86.3470 %