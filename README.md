# Face Aging with Identity-Preserved Conditional Generative Adversarial Networks
The repo refers to the official open source of paper [Face Aging with Identity-Preserved Conditional Generative Adversarial Networks]

While their instruction is not comperhansive leading to it is hard to run the code directly following their instruction. In order to run the code smoothly for research and study, some parts of codes are modified, and add three tools for generating source files, finlaly the code has ungraded to Tensorflow 1.14.1.

Please follow the instructions to prepare and run the programme.
## Precondition for data: image name content should have age label e.g.: "14_0_4_Aaron_Johnson_0001.jpg": 14 is age and should include symbol '_'.


# 1. Prepared source files (path: ./tools/)for programme execution
1) sourcefile.txt
2) tain_data 

 (file content: image name and age)

Optional:
train_label_pair.txt

2. Test
1) pre-trained model for test
python pre_trainedmodel_test.py

2) customer model for test
python customer_test.py --customer_model_number=199999

2 Traing from scratch
python age_lsgan_transfer.py \
  --gan_loss_weight=75 \
  --fea_loss_weight=0.5e-4 \
  --age_loss_weight=30 \
  --fea_layer_name=conv5 \
  --learning_rate=1e-3 \
  --batch_size=32 \
  --image_size=128 \
  --max_steps=500000
  

## Reference
https://github.com/dawei6875797/Face-Aging-with-Identity-Preserved-Conditional-Generative-Adversarial-Networks.git