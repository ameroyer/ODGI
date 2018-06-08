## README

This is the official Tensorflow implementation for `ODGI: Object Detection with Grouped Instances`.


The library requirements are:

  * Tensorflow (1.4)
  * Python (3.5)
  * Numpy

### Model
Current state-of-the-art detection systems often suffer of two important shortcomings: processing speed and detecting objects at varying scales. In this project, we propose ODGI (*Object Detection with Grouped Instances*) a new detection scheme that addresses these issues; The main idea is to allow the detector to predict groups of objects rather than individuals, when it is needed. The proposed model allows working at lower resolution, thereby saving computations, and that the ability to identify groups leads to fewer, yet more meaningful, regions proposal than existing methods.

![ODGI overview](readme_images/model.png)

### Datasets
In ``Data`` we provide pre-computed files used by the input data pipeline for the [VEDAI](https://downloads.greyc.fr/vedai/) dataset and the [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/). For each dataset, we provide:

  * `{}_{train, test}` are TFRecords containing the ground-truth labels. Each example has features `im_id` (an image id that we use to resolve the path of the corresponding image), `num_boxes` (number of valid bounding boxes), `bounding_boxes` and `classes`.
  * `metadata_{}` contains paths to the train and test TFRecords for this dataset, path to the main image folder and other information (number of samples, etc.)
  * `{}_split{train, test}` contains the image IDS for the train and test split we used in our experiments
 
![SDD inputs](readme_images/sdd_inputs.png)
 
Additionally you'll need to have the images stored in some `image_folder` that you can modify in `metadata_{}`.
See the notebook `input_pipeline.ipynb` for how the images and annotations TFRecords are generated.

### Train the model

We provide notebooks `train_standard.ipynb` to train and evaluate  a standard `tiny-yolov2` model, and `train_odgi.ipynb` to train and evaluate a two-stage ODGI pipeline.
Each notebook contains a configuration set-up and build the Tensorflow graph for both training and evaluation using functions defined in `net.py`, `loss_utils.py` and `eval_utils.py`.
The default configuration options and short descriptions can be found in `defaults.py`.

Most of the training process can be monitored via Tensorboard (the default output directory is `./log`). In particular we output the following summaries:

  * **[text]** `config_summary` contains all configuration options for the current run.
  * **[scalars]** We report the training losses (`train_tinyyolov2` for standard and `train_stage1` and `train_stage2` for ODGI).
  We also report running evaluation metrics. In particular the final detection metrics for both models are respectively `eval/tinyyolov2_avgprec_*` for standard and `eval/stage2_avgprec_*` for ODGI.
  * **[images]** Image summaries contain 
     * image *inputs* (vizualized with ground-truth bounding boxes, and empty cells at lower opacity), 
     * during training it contains the predicted *boxes assigned* to the ground-truth (before and after rescaling in the ODGI setting)
     * the *output bounding boxes* above a certain confidence threhsold (default is 0.5)
     * extracted *crops* after intermediate ODGI stages
     * group flag *confusion matrix*
     
Note that in the ODGI case, durig training the two stages are trained independently hence summaries can also be read independently. While for evaluation we always test the full pipeline hence we have an additional summary that contains the boxes predicted by the last stage of the pipeline merged with the ones kept back at earlier stages.


### Launch a pre-trained model

`load_and_eval` is a small example of how to load a pretrained model (ODGI or standard) and compute detection metrics on a given dataset as well as output the resulting images. 

Note that it uses the default graph configuration, in order to customize it (e.g., to change the number of extracted crops etc.), it is easier to modify `train_{odgi, standard}` by removing the train graph part, loading an existing model weights and customizing the configuration.
