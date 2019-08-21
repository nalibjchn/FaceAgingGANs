#### General data analysis

The experiment on the Google Colab, the aim is general checking the age and gender distribution in three datasets, UTKFace, IMDB_WIKI and CACD2000, however,
Analysis_DATASET(IMDBWIKI+UTKFace+CACD2000).ipynb

#### Face alignments for data processing 

1. Precondition: 
    Install dependencies.
```

      pip install imutils 
      pip install dlib

```

2. Run programme with custom setting
```
   python align_faces.py \
       --root_dir "../DATA/" \  # directory to source data
       --des_dir "training_align" \ # directory to destination for aligned data
       --notdetect_dir "notdetect" # directory to dir

```

#### Reference
1. https://github.com/contail/Face-Alignment-with-OpenCV-and-Python.git
2. https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
