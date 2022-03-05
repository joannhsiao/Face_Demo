export MXNET_CPU_WORKER_NTHREADS=8
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
#rm output.txt
mkdir backup_result
mv *.csv backup_result
mv *.jpg backup_result
mkdir backup_model
mv *.json backup_model
mv *.params backup_model
python Face_efm_v3.py /lab307/triplet_8398 > output.txt 2>&1 & 
