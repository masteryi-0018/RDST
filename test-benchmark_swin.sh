model_1='H:\spyder\liif-main\save\_train_swin-liif\epoch-best.pth'
gpu_2='0'

echo 'set5' &&
echo 'x2' &&
python test_swin-liif.py --config ./configs/test-swin/test-set5-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_swin-liif.py --config ./configs/test-swin/test-set5-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_swin-liif.py --config ./configs/test-swin/test-set5-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_swin-liif.py --config ./configs/test-swin/test-set5-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_swin-liif.py --config ./configs/test-swin/test-set5-8.yaml --model $model_1 --gpu $gpu_2 &&

echo 'set14' &&
echo 'x2' &&
python test_swin-liif.py --config ./configs/test-swin/test-set14-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_swin-liif.py --config ./configs/test-swin/test-set14-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_swin-liif.py --config ./configs/test-swin/test-set14-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_swin-liif.py --config ./configs/test-swin/test-set14-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_swin-liif.py --config ./configs/test-swin/test-set14-8.yaml --model $model_1 --gpu $gpu_2 &&

echo 'b100' &&
echo 'x2' &&
python test_swin-liif.py --config ./configs/test-swin/test-b100-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_swin-liif.py --config ./configs/test-swin/test-b100-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_swin-liif.py --config ./configs/test-swin/test-b100-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_swin-liif.py --config ./configs/test-swin/test-b100-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_swin-liif.py --config ./configs/test-swin/test-b100-8.yaml --model $model_1 --gpu $gpu_2 &&

echo 'urban100' &&
echo 'x2' &&
python test_swin-liif.py --config ./configs/test-swin/test-urban100-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_swin-liif.py --config ./configs/test-swin/test-urban100-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_swin-liif.py --config ./configs/test-swin/test-urban100-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_swin-liif.py --config ./configs/test-swin/test-urban100-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_swin-liif.py --config ./configs/test-swin/test-urban100-8.yaml --model $model_1 --gpu $gpu_2 &&

true
