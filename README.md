# Training command
```
nohup python -m train.train_amp --dataset=cifar10 --model=raft --encoder=resnet18 --projector=byol-proj --predictor=byol-proj --mlp-middim=512 --mlp-outdim=256 --normalization=l2 --gpus=4 --port=8010 --num-workers=1 --epochs=300 --batch-size=1024 --rand-seed=2333 --resize-dim=32 --optimizer=adam --lr=3e-4 --ema-lr=4e-3 --ema-mode=sgd --checkpoint-epochs=10 --checkpoint-dir=checkpoints/byol &
```
