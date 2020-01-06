# Region Proposal Network

This is simple RPN implementation using tensorflow. It can be used for Faster-RCNN. Most of the operations performed during the implementation were carried out as described in the [paper](https://arxiv.org/pdf/1506.01497.pdf). However, unlike the vgg16 model used in paper, the last max pooling layer was not removed from the model and the stride value was used as **32**. Also, positive and negative anchor number arranged as **64**, which was set to **128** in the paper. As the training and tests were performed on the [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset, the anchor scale values were applied as **[64, 128, 256]**.  

## Usage

Project models created in virtual environment using [miniconda](https://docs.conda.io/en/latest/miniconda.html).
You can also create required virtual environment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

To create virtual environment (tensorflow-2 gpu environment):

```sh
conda env create -f environment.yml
```

To train your model:

```sh
python trainer.py
```

If you have gpu issues you can use **-handle-gpu** flag:

```sh
python trainer.py -handle-gpu
```

To predict using your model:

```sh
python predictor.py
```

You can also use **-handle-gpu** flag like on the training.


Grid - center of the every anchor boxes
![Grid](http://furkanomerustaoglu.com/wp-content/uploads/2019/12/grid_map.png)

Anchors output_width * output_height * anchor_count
![Anchors](http://furkanomerustaoglu.com/wp-content/uploads/2019/12/anchors.png)

The model was trained with the [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) dataset. Although the dataset did not contain any image of a lion, it was also able to identify the lion through images of cats and other animals used during training.
![Prediction](http://furkanomerustaoglu.com/wp-content/uploads/2020/01/prediction.png)

Photo by Werner van Greuning on Unsplash
