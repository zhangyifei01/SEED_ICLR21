export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_seed.py \
  -ta resnet50 -sa resnet18 \
  --lr 0.03 \
  --batch-size 256 --seed-k 65536\
  --mlp --seed-st 0.2 --seed-tt 0.01 --aug-plus --cos \
  --tpretrained ../checkpoint_0099.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data1/ILSVRC2012

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_lincls.py \
  -a resnet18 \
  --lr 30.0 \
  --batch-size 256 \
  --pretrained checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  /data1/ILSVRC2012

