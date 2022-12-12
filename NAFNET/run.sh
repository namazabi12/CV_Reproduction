##pip install basicsr
#apt-get update -y
#apt-get install apt-file -y
#apt-get install libsm6 -y
#apt-get install libxrender1 -y
#apt-get install libxext6 -y
#pip install -r requirements.txt
#nvidia-smi
#python3 test.py
#conda env list
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
source activate run
#python scripts/data_preparation/sidd.py
#python basicsr/train.py -opt options/train/EDSR/train_EDSR_Lx2.yml
python basicsr/train.py -opt options/train/NAFNET/train_NAFNET_Tx2.yml
#python scripts/data_preparation/sidd.py



