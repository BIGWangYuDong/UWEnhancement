# Useful Tools 

Most useful tools are in [tools](../tools/) folder

## Visualize Network Structure

You can use [visualize_network.py](../tools/visualize_network.py) to check your network structure 

The method of using the file is as follows:

1. init network layers in function _init_layers
2. add parameters in cfg_model if need
3. run:

```
cd tools/
python visualize_network.py
tensorboard --logdir runs/XXX 			# XXX means folder name
```
Usually you can see the network structure at http://localhost:6006/#

## Train Process Monitoring and Training Details Review

You can always know the details while traning (usually at http://localhost:8097/). We currently support visualize loss and image during training process. But at present, we only support manual addition loss and image.

The method of adding loss and image visualization are as follows:
```
losses['loss_XXX'] = loss_XXX.data.cpu()		# adding loss visualization

XXX = normimage(XXX, save_cfg=save_cfg) 		# adding image visualization
shows.append(XXX.transpose([2, 0, 1]))

```


Every training process we save a new log file and you can see the option and training details from this log file. Also, you can use tensorboard to visualize the loss curve:
```
cd XXXX							# workdir root
tensorboard --logdir tf_logs/XXX			# XXX means file name
```
## Test Time Argument x8 

You can ues [test_TTAx8.py](../test_TTAx8.py) to use Test Time Argument(TTA)x8. Testing way is same as test.py

## Tensorflow to PyTorch

You can use [tf2torch.py](../tf2torch.py) to convert Tensorflow checkpoint to PyTorch checkpoint, which can be loaded by PyTorch Network.

The method of using the file is as follows:
1. change label name (checkpoint dir name)
2. init pytorch based network, it is important that the torch conv layer name should same as tf name!
3. change save path root.
4. run.

## Output Feature Map

You can use [output_featuremap.py](../tools/output_featuremap.py) to output the feature map of network middle layer.
The method of using the file is as follows:
1. change args.config and args.load_from root
2. add the layer name in name_list
3. change the output root and image root
4. run 
```
python output_featuremap.py
```
## Get WaterNet Data

Copy from Water-Net-Code, it's a MATLAB code, you can use [generate_test_data.m](../tools/get_waternet_data/generate_test_data.m) to obtain Histogram Equalization(HE), Gamma Correction(GC) and White Balance(WB) processed image.








