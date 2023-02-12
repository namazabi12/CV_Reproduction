source activate run


#python basicsr/train.py -opt options/train/EDSR/train_EDSR_Mx2.yml
#python basicsr/train.py -opt options/train/EDSR/train_EDSR_HOAF_Mx2.yml
#python basicsr/train.py -opt options/train/EDSR/train_EDSR_NAF_Mx2.yml
python basicsr/train.py -opt options/train/EDSR/train_EDSR_NAF_HOAF_Mx2_2.yml


#python basicsr/test.py -opt options/test/EDSR/test_EDSR_Mx2.yml
#python basicsr/test.py -opt options/test/EDSR/test_EDSR_HOAF_Mx2.yml
#python basicsr/test.py -opt options/test/EDSR/test_EDSR_NAF_Mx2.yml
#python basicsr/test.py -opt options/test/EDSR/test_EDSR_NAF_HOAF_Mx2.yml