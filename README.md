# PCT: Point Cloud Transformer

This repository contains a PyTorch implementation of PCT: Point Cloud Transformer, designed for the segmentation and classification of 3D point cloud data.

## Motivation and Objective
The project aims to leverage the Transformer architecture, traditionally dominant in NLP, to process unordered 3D point cloud data, facilitating tasks such as augmented reality, autonomous driving, and robotics. By treating each point as a token, we explore transformer networks for vision tasks, generate a synthetic point cloud data using domain randomization, and implement the state-of-the-art Point Cloud Transformer (PCT).

## Synthetic Data Generation
The Transformer architecture requires a large amount of training data. Given that point cloud data is not always readily available, we generated synthetic point cloud data using NVIDIA Omniverse Replicator to simulate dent detection on sheet metals, commonly used in the automotive industry.

## Requirements
- Python >= 3.7
- PyTorch >= 1.6
- h5py
- scikit-learn


## Training and Testing:

### Train
```
python main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size=32 --epochs=250 --lr=0.0001
```

### Test
```
python main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size=8
```

## Experimental Results
We evaluated the PCT on the ModelNet40 dataset, a widely used benchmark for shape classification with 12,311 CAD models in 40 object categories. The training used cross-entropy loss and the SGD optimizer with momentum 0.9, batch size 32, and 250 epochs.

### Training Results

| Learning Rate	|   Train Accuracy (%)	|   Test Accuracy (%) |
|-----------------|-----------------------|---------------------|
|      0.01	      |       exploded	      |      exploded       |
|      0.005	|        50.41	      |      33.95          |
|      0.001	|        94.88	      |      91.41          |
|      0.0005	|        95.56	      |      91.89          |
|      0.0001	|        99.27	      |      92.46          |


## Conclusion
The Transformer architecture, with its invariance towards cardinality and permutation, is well-suited for unordered datasets like point clouds with irregular domains. However, given its data-hungry nature, synthetic point cloud data generation is crucial for training when real datasets are limited.


### Citation

```latex
@misc{guo2020pct,
      title={PCT: Point Cloud Transformer}, 
      author={Meng-Hao Guo and Jun-Xiong Cai and Zheng-Ning Liu and Tai-Jiang Mu and Ralph R. Martin and Shi-Min Hu},
      year={2020},
      eprint={2012.09688},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
