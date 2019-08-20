# Study of Cycle-consistent Generative Adversarial Networks (CycleGANs) for face aging
The repo refers to the official open source of paper [Cycle-consistent Generative Adversarial Networks (CycleGANs)] and other related CycleGANs source codes by Keras Framework.

As a source of research and study for dissertation, firstly, quick implementation in Google Colab (refer to "CycleGAN_v1"), and a comperhasive source code including training, testing and custom parameters setting in "CycleGAN_v2"

Please follow the instructions to prepare and run the programme.

## 1. Traing

1) Programme CycleGAN_v1
```
* Run the scripts without custom parameters

  cd CycleGAN_v1
  python faceaging_cyclegan.py
```

2) Training from scratch

```
* Run the scripts with custom parameters.
  python main.py \
    --is_train True \
    --epoch 50 \
    --dataset ../DATA/TrainingSet_CACD2000 \
    --savedir save \
    --use_trained_model False \
    --use_init_model False
```
**NOTE**: During the training process, the "checkpoint", "samples", "summary" and "test" folders will be created in the "save" folder automatically. 
 - "checkpoints": save trained model.
 - "samples": save the test images( 100 images in a one png files (10*10), and the top 10 images on the first rom in one sample images will be fed into intermediate trained model. 
 - "summary": save loss values by TensorBroad.
 - "test": save the generated results each epoch.

## 3. Test 
```
    python main.py \
    --is_train False \
    --testdir test_image_dir \ # default: test
    --savedir save 
```
**NOTE**:
   the test result will be saved into './save/test' folders, which are two png images by genders. "test_as_female.png" and "test_as_male.png"

## 4. Experiment result
 - Female (Left) and Male (Right) results:
<p align="center">
  <img src="save/test/test_as_female.png" height="400",width="800">|||
  <img src="save/test/test_as_male.png" height="400",width="800">
</p>

## 5. Files
* [`ops.py`](ops.py): Build layers, such as convoluaiton, fully connection, activation function(Leaky ReLU), and images operation (load and save images).
* [`FaceAging.py`](FaceAging.py): a class, to build a model by calling 'ops.py'
* [`main.py`](main.py): start the programme to run `FaceAging.py`.

## Reference
- [CycleGANs tutorial] https://hardikbansal.github.io/CycleGANBlog/
- [CycleGANs Offical Open source code, TensorFlow] https://github.com/junyanz/CycleGAN.git
- [CycleGANs Offical Open source code, Keras] https://github.com/simontomaskarlsson/CycleGAN-Keras.git
-  https://github.com/eriklindernoren/Keras-GAN.git
