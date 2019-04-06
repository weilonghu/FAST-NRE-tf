# FAST-NRE-tf

#### Introduction
FAST neural relation extraction. We have implemented several methods of neural relation extraction. This project aims to help beginners of relation extraction start experiments quickly. The source code is referenced but not limited to [Lin et.al (2016)](https://github.com/thunlp/OpenNRE)。

#### Requirements

1. Python（2 or 3）
2. Numpy
3. Tensorflow （>=1.4)
4. sklearn

#### Usage

We also split the network into four layers: **Embedding** layer，**Encoder** layer，**Selector** layer and **Classifier** layer. Among them，Encoder layer and Selector layer are the main parts we want to extend. The role of each layer is as follows：

1. Embedding-layer：trainsform text to word embedding and position embedding...
2. Encoder-laye: use CNN or PCNN to extract features of sentences.
3. Selector-layer：use selective attention to calculate weights for each sentence，then produce the bag feature
4. Classifier-layer：use cross-entropy to calculate loss of the network.

##### Code Description

- `init_data.py`: use this script to convert plain text into numpy format.
- `engine.py`: who controls the process of trainging and testing.
- `main.py`: the entrypoint of the project, initailize the arguments.
- `model.py`: integrate all layers.
- `utils.py`: contains codes for common use.

##### Run Description

Before you run the model for the first time, you should initialize the dataset:

```bash
python main.py --is_train=True --preprocess=True [--clean=True]
```

The`--clean`option determines whether to clear existing model files. After it, there is no need to initialize the data again.

Run the model:

```bash
python main.py --is_train=True
```

Test the model：

```bash
python main.py --is_train=False
```

#### Dataset

Our dataset are based on Lin et.al.  but we keep the entity types. The following list are all files：

1. relation2id.txt：relation mapping file
2. type2id.txt：entity type mapping file
3. vec.txt：word2vec file
4. train.txt：training set
5. test.txt：testing set

The dataset can be download from [here](https://pan.baidu.com/s/1C-z_v-PivAlQvmL9S3ySkg).  The format of data is as follows：

```bash
entity1_mid entity2_mid entity1 entity2 entity1_type entity2_type relation_label sentence ###END###
```


#### References

1. **Neural Relation Extraction with Selective Attention over Instances.** *Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun.* ACL2016. [paper](http://www.aclweb.org/anthology/P16-1200)
2. **Distant supervision for Relation Extraction via Piecewise Convolutional Neural Networks.** Zeng D, Liu K, Chen Y, et al. EMNLP2015. [paper](http://www.aclweb.org/anthology/D15-1203)