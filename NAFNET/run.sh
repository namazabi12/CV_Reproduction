#pip install basicsr
apt-get update -y
apt-get install apt-file -y
apt-get install libsm6 -y
apt-get install libxrender1 -y
apt-get install libxext6 -y
#pip install -r requirements.txt
python basicsr/train.py -opt options/train/EDSR/train_EDSR_Lx2.yml
exec /bin/bash