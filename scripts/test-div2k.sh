model_1='H:/spyder/RDST_model/_train_edsr-baseline-liif\epoch-last.pth'
gpu_2='0'

echo 'div2k-x2' &&
python test_liif.py --config ./configs/test/test-div2k-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'div2k-x3' &&
python test_liif.py --config ./configs/test/test-div2k-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'div2k-x4' &&
python test_liif.py --config ./configs/test/test-div2k-4.yaml --model $model_1 --gpu $gpu_2 &&

echo 'div2k-x6*' &&
python test_liif.py --config ./configs/test/test-div2k-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'div2k-x12*' &&
python test_liif.py --config ./configs/test/test-div2k-12.yaml --model $model_1 --gpu $gpu_2 &&
echo 'div2k-x18*' &&
python test_liif.py --config ./configs/test/test-div2k-18.yaml --model $model_1 --gpu $gpu_2 &&
echo 'div2k-x24*' &&
python test_liif.py --config ./configs/test/test-div2k-24.yaml --model $model_1 --gpu $gpu_2 &&
echo 'div2k-x30*' &&
python test_liif.py --config ./configs/test/test-div2k-30.yaml --model $model_1 --gpu $gpu_2 &&

true
