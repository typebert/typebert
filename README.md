<img src="TypeBert.svg" width="800" alt="header"/>

This is the source code repository for the ESEC/FSE paper [*Learning Type Annotation: Is Big Data Enough?*](https://www.kevinrjesse.com/pdfs/typebert_esec_fse_.pdf) 

It is derived from Tensorflow model garden.

## Docker:
We have provided a docker image that will have all dependencies install for TypeBert. By downloading the image, you can link the directory of with the model
weights through the following steps.

1. Install docker if you don't have it. [Docker](https://www.docker.com/get-started)

2. Pull the latest TypeBert image ```docker pull typebert/typebert```

3. Run docker with the code directory. Model weights can also be placed in this folder and passed via command line to the `run_pretraining.py` or `run_classifier.py`. To run docker with NVidia GPUs, please use [CUDA 11](https://developer.nvidia.com/cuda-downloads) and [cuDNN 8](https://developer.nvidia.com/cudnn). We use the latest and greatest TensorFlow 2.4. Eventually, in order to run your docker use docker run i.e. 
```docker run --gpus all --rm --mount type=bind,src=/home/typebert,dst=/home/typebert --mount type=bind,src=/data2/typebert_data,dst=/data2/typebertv2_data -it typebert/typebert bash```We choose to keep the typebert code on a data drive. Mounting the directory makes it visible to the docker container. After cloning the code in the following step, make sure to export the model directory in the python path 

`export PYTHONPATH=$PYTHONPATH:/home/myhome/TypeBert/models/`


## Code
Clone the code from this repository with git cli. This can be done in the docker or in a binded directory. 
`git clone https://github.com/typebert/typebert.git`

## Data
We have uploaded compressed directories of the code datasets. These data directories are made up of multiple smaller tf_record files. Each folder contains the same sentence piece tokenizer model trained on typescript. All sentencepeice related files are prefixed "ts_sp". 

### Pretraining:
* Javascript Corpus: [link](https://drive.google.com/uc?export=download&id=1Cq_K1QBqoRDv_gLpz3dKBQYbHO_CJbJm)
1. First download the javascript corpus. Extact the .tgz file and add it to a your visible folder.
`tar -xvzf pretrain.tgz `


### FineTuning:
* TypeBert Type Data: [link](https://drive.google.com/uc?export=download&id=10Kw7PyoVMQC_hwKzhHFezNOpCWBg3b3a)

## Model Weights:
* Pretrained Weights: [link](https://drive.google.com/uc?export=download&id=1Do1b2AB_unTeyi0Dbx8DEdwINT2OIFf5)
* FineTuning Weights on Type Dataset [link](https://drive.google.com/uc?export=download&id=1g8mm2aeBYG_K3U4O9WxE5A11mh7H7j5N)

## Running evaluation on TypeBert data
In your docker, go to your TypeBert/type-bert directory. To run fine-tuning, it would look something like this. The number of training batch is dependent on how many GPUs you use. Heuristically, it was found that 25 per GPU will work.

### How to evaluate models
`python run_classifier.py --mode='predict' --input_meta_data_path=/data2/typebertv2_data/meta_data --train_data_path=/data2/typebertv2_data/train.tf_record --eval_data_path=/data2/typebertv2_data/test.tf_record --bert_config_file=/home/myhome/TypeBert/type-bert/bert_config.json --predict_checkpoint_path=/home/myhome/typebert_my_model/ckpt-272457 --train_batch_size=32 --eval_batch_size=32  --model_dir=/home/myhome/typebert_my_model/ --distribution_strategy=multi_worker_mirrored`

This method will return test labels & (indices values numpy files). The test labels are the most useful for computing stats like top 1 accuracy. However top 5 accuracy for example requires the probabilities for each class. We report the top 100 probablities for each prediction as anything greater than that is not particularly useful for these metrics and make the files intractably large.

### How to Fine Tune
Fine-tuning is the process of refining the TypeBert model weights for the type inference or sequence tagging task. It starts with a pretrained model on JavaScript and tunes it to TypeScript.

`python run_classifier.py mode='train_and_eval' --input_meta_data_path=/data2/typebertv2_data/meta_data --train_data_path=/data2/typebertv2_data/train.tf_record --eval_data_path=/data2/typebertv2_data/test.tf_record --bert_config_file=/home/myhome/TypeBert/type-bert/bert_config.json --init_checkpoint=/data2/typebert_pretrain_weights/bert_model_step_100000.ckpt-10 --train_batch_size=100 --eval_batch_size=100 --steps_per_loop=1 --learning_rate=2e-5 --num_train_epochs=4 --model_dir=/home/myhome/typebert_my_model --distribution_strategy=multi_worker_mirrored`

### How to Pretrain From scratch
Pretrain a BERT architecture for JavaScript. This uses an MLM and NSP task to refine the models prediction of the next token. We have done this costly step for you
and uploaded the model weights. 

`python run_pretraining.py --input_files=/data2/myhome/pretrain/*.tfrecord --bert_config_file=/home/myhome/TypeBert/type-bert/bert_config.json --distribution_strategy=multi_worker_mirrored --model_dir=/home/myhome/my_typebert_pretrain/ --num_gpus=6 --train_batch_size=100`
