model_1='H:/spyder/RDST_model/_train_edsr-baseline-liif\epoch-last.pth'
gpu_2='0'

echo 'set5' &&
echo 'x2' &&
python test_liif.py --config ./configs/test/test-set5-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_liif.py --config ./configs/test/test-set5-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_liif.py --config ./configs/test/test-set5-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_liif.py --config ./configs/test/test-set5-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_liif.py --config ./configs/test/test-set5-8.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x12*' &&
python test_liif.py --config ./configs/test/test-set5-12.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x18*' &&
python test_liif.py --config ./configs/test/test-set5-18.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x24*' &&
python test_liif.py --config ./configs/test/test-set5-24.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x30*' &&
python test_liif.py --config ./configs/test/test-set5-30.yaml --model $model_1 --gpu $gpu_2 &&

echo 'set14' &&
echo 'x2' &&
python test_liif.py --config ./configs/test/test-set14-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_liif.py --config ./configs/test/test-set14-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_liif.py --config ./configs/test/test-set14-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_liif.py --config ./configs/test/test-set14-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_liif.py --config ./configs/test/test-set14-8.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x12*' &&
python test_liif.py --config ./configs/test/test-set14-12.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x18*' &&
python test_liif.py --config ./configs/test/test-set14-18.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x24*' &&
python test_liif.py --config ./configs/test/test-set14-24.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x30*' &&
python test_liif.py --config ./configs/test/test-set14-30.yaml --model $model_1 --gpu $gpu_2 &&

echo 'b100' &&
echo 'x2' &&
python test_liif.py --config ./configs/test/test-b100-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_liif.py --config ./configs/test/test-b100-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_liif.py --config ./configs/test/test-b100-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_liif.py --config ./configs/test/test-b100-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_liif.py --config ./configs/test/test-b100-8.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x12*' &&
python test_liif.py --config ./configs/test/test-b100-12.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x18*' &&
python test_liif.py --config ./configs/test/test-b100-18.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x24*' &&
python test_liif.py --config ./configs/test/test-b100-24.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x30*' &&
python test_liif.py --config ./configs/test/test-b100-30.yaml --model $model_1 --gpu $gpu_2 &&

echo 'urban100' &&
echo 'x2' &&
python test_liif.py --config ./configs/test/test-urban100-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_liif.py --config ./configs/test/test-urban100-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_liif.py --config ./configs/test/test-urban100-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_liif.py --config ./configs/test/test-urban100-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_liif.py --config ./configs/test/test-urban100-8.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x12*' &&
python test_liif.py --config ./configs/test/test-urban100-12.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x18*' &&
python test_liif.py --config ./configs/test/test-urban100-18.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x24*' &&
python test_liif.py --config ./configs/test/test-urban100-24.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x30*' &&
python test_liif.py --config ./configs/test/test-urban100-30.yaml --model $model_1 --gpu $gpu_2 &&

echo 'manga109' &&
echo 'x2' &&
python test_liif.py --config ./configs/test/test-manga109-2.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x3' &&
python test_liif.py --config ./configs/test/test-manga109-3.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x4' &&
python test_liif.py --config ./configs/test/test-manga109-4.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x6*' &&
python test_liif.py --config ./configs/test/test-manga109-6.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x8*' &&
python test_liif.py --config ./configs/test/test-manga109-8.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x12*' &&
python test_liif.py --config ./configs/test/test-manga109-12.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x18*' &&
python test_liif.py --config ./configs/test/test-manga109-18.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x24*' &&
python test_liif.py --config ./configs/test/test-manga109-24.yaml --model $model_1 --gpu $gpu_2 &&
echo 'x30*' &&
python test_liif.py --config ./configs/test/test-manga109-30.yaml --model $model_1 --gpu $gpu_2 &&

true