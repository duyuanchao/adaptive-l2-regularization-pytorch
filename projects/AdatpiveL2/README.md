# Adatpive L2 Regularization

This repo implement Adaptive L2 Regularization in Person Re-Identification, with pytorch (fast-reid).


## Performance 

Result on Market-1501
| Method | Pretrained | Rank@1 | mAP | mINP |
| :---: | :---: | :---: |:---: | :---: |
| BoT(R50) | ImageNet | 94.18% | 85.01% | 57.23% |
| BoT(R50)-AdaptiveL2 | ImageNet | 94.71%(+0.53%) | 86.33%(+1.33%) | 59.77%(+2.54%) |

## Train
```
python3 train_net.py --config-file ./configs/adaptive_l2_bot.yaml --num-gpus 4
```

## Evaluation 
```
python3 train_net.py --config-file ./configs/adaptive_l2_bot.yaml --eval-only MODEL.WEIGHTS /mnt/tensorboard/model_best.pth MODEL.DEVICE "cuda:0"
```

## Acknowledgement

- [fast-reid](https://github.com/JDAI-CV/fast-reid)
- [AdaptiveL2Regularization](https://github.com/nixingyang/AdaptiveL2Regularization)

