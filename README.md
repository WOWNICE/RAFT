# Shell Command
### Train
CIFAR10(0):
```
CUDA_VISIBLE_DEVICES=0,1 nohup python -m train.train --dataset=cifar10 \
--model=raft --encoder=resnet18 --projector=byol-proj --predictor=linear-proj \
--mlp-middim=512 --mlp-outdim=256 --normalization=l2 --gpus=2 --port=8010 --num-workers=1 \
--epochs=1000 --batch-size=512 --rand-seed=2333 --resize-dim=32 --optimizer=adam --lr=3e-4 \
--ema-lr=4e-3 --ema-mode=sgd --checkpoint-epochs=50 --amp=False --sync-bn=False \
--checkpoint-dir=checkpoint/path/ --wrapper=weight_wrapper.EmaWrapper &
```

ImageNet:
```
CUDA_VISIBLE_DEVICES=2,3 nohup python -m train.train --dataset=imagenet \
--model=raft --encoder=resnet18 --projector=byol-proj --predictor=byol-proj --mlp-middim=512 \
--mlp-outdim=256 --normalization=l2 --gpus=2 --port=8011 --num-workers=1 --epochs=1000 \
--batch-size=12 --rand-seed=2333 --resize-dim=224 --optimizer=adam --lr=3e-4 --ema-lr=4e-3 \
--ema-mode=sgd --checkpoint-epochs=50 --amp=False --sync-bn=False --checkpoint-dir=checkpoints/path/ &
```

### Eval
CIFAR10(0)
```
nohup python -m eval.eval --dataset=imagenet --encoder=resnet18 --gpu=3 --epochs=200 \
--batch-size=40 --resize-dim=32 --eval-mode=online --checkpoint=checkpoint/path/ &
```

ImageNet
```
nohup python -m eval.eval --dataset=imagenet --encoder=resnet18 --gpu=3 --epochs=200 \
--batch-size=40 --resize-dim=224 --eval-mode=online --checkpoint=checkpoint/path/ &
```
