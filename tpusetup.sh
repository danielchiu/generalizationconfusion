conda activate torch-xla-1.5
export TPU_IP_ADDRESS=10.172.245.162
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
