export MXNET_CPU_WORKER_NTHREADS=8
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
mkdir backup_result
mv *.csv backup_result
mv *.jpg backup_result
mkdir backup_model
mv *.json backup_model
mv *.params backup_model

python Face_efm_v3.py /lab307/Celeb1M > output.txt 2>&1 &
