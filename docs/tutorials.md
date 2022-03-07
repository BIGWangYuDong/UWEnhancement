# Tutorial

We use registration mechanism to build model, data, dataset, optimizer, scheduler, loss function etc. The main code are all in [core](../core) folder.

## Customize Models

We basically categorize model components into **backbone** and **Model(Network)**.

### Develop new components

#### I. Add a new Model:

Here we show how to develop new Model(Network) with an example of  AANet.

**1. Define a new model(Network)**

Create a new file ``` core/Models/aanet.py```

```python
import torch
import torch.nn as nn
from UW.core.Models.builder import NETWORK, build_backbone
from UW.core.Models.base_model import BaseNet
from UW.core.Models.weight_init import normal_init, xavier_init

@NETWORK.register_module()
class AANet(BaseNet):
	 def __init__(self,
                 backbone,
                 pretrained=None,
                 init_weight_type=None,
                 get_parameter=True):
        super(AANet, self).__init__(backbone, pretrained, init_weight_type, get_parameter)
         if backbone is not None:
            self.backbone = build_backbone(backbone)
        else:
            self.backbone = None
        if init_weight_type is not None:
            self.init_weight_type = init_weight_type
        self.get_parameter = get_parameter
        self._init_layers()
        self.init_weight(pretrained=pretrained)
        self.get_parameters()
        
    def _init_layers(self)
    	'''
    	init layers
    	'''
        pass
    
    def init_weight(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m)
        if self.backbone is not None:
            self.backbone.init_weights(pretrained)
    
    def forward(self, x)
    	pass
```

**2. Import the module**

add the following line to ```core/Models/__init__.py```	

```python
from .aanet import AANet
```

**3. Use the model(Network) in your config file.**

```python
model = dict(
    type='AANet',
    arg1=XXX,
    arg2=XXX,
    ...)
```

#### II. Add a new backbone

**1. Define a new backbone**

Create a new file ``` core/Models/backbone/aabackbone.py```

```python
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
from UW.core.Models import BACKBONES
from UW.core.Models.weight_init import normal_init

@BACKBONES.register_module()
class AABackbone(nn.Module):
    def __init__(self, pretrained, ...):
        pass
    
    def forward(self, x):
        pass
    
    def init_weights(self, pretrained=None):
        pass
```

**2. Import the module**

add the following line to ```core/Models/backbone/__init__.py```	

```python
from .aabackbone import AABackbone
```

**3. Use the model(Network) in your config file**

```python
model = dict(
    type=...,
    backbone=dict(type='AABackbone', 
                  pretrained=True, 			# or False
                  arg1=xxx,
                  ...)
    ...)
```

#### III. Add a new Pipeline:

**1. Define a new data pipeline**

Create a new file ``` core/Datasets/Pipelines/trans.py```

```python
from UW.core.Datasets.builder import PIPELINES

@PIPELINES.register_module()
class Pipeline(object):
    def __init__(self, arg1, ...):
        pass
    
    def __call__(self, results):
        pass
```

**2. Import the module**

add the following line to ```core/Datasets/Pipelines/__init__.py```	

```python
from .trans import Pipeline
```

**3. Use the pipeline in your config file**

```python
train_pipeline = [...,
                  dict(type='Pipeline', arg1=xxx, ...),
    			  ...]
# OR
test_pipeling = [...,
                 dict(type='Pipeline', arg1=xxx, ...),
                 ...]
```

#### IV. Add a new Loss function:

**1. Define a new loss function**

Create a new file ``` core/Losses/a_loss.py```

```python
import torch
import torch.nn as nn
from UW.core.Losses.builder import LOSSES

@LOSSES.register_module()
class ALoss(nn.Module):
    def __init__(self, loss_weight=1.0, ...):
        self.loss_weight = loss_weight
        pass
    
    def forward(self, img1, img2):
        '''
        loss = loss * self.loss_weight
        '''
        pass
```

**2. Import the module**

add the following line to ```core/Losses/__init__.py```	

```python
from .a_loss import ALoss
```

**3. Use the loss function in your config file**

```python
loss_a = dict(type='ALoss', loss_weight=xxx, arg1=xxx,...)
```

#### V. Add a new Dataset:

We are going to support unaligned dataset, if you need another loading dataset way, you can add a new Dataset by:

**1. Define a new dataset

Create a new file ``` core/Datasets/a_dataset.py```

```python
from UW.core.Datasets.base_dataset import BaseDataset
from UW.core.Datasets.builder import DATASETS
from UW.core.Datasets.Pipelines import Compose
import copy

@DATASETS.register_module()
class ADataset(BaseDataset):
    def __init__(self, **kwargs):
        super(AlignedDataset, self).__init__(**kwargs)
        pass
    
    def prepare_train_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)
    
    def xxx(self, arg1, arg2, ...):
        pass
```

**2. Import the module**

add the following line to ```core/Datasets/__init__.py```	

```python
from .a_dataset import ADataset
```

**3. Use the dataset in your config file**

```python
dataset_type = 'ADataset'
data = dict(
    ...
    train=dict(
        type=dataset_type,
        arg1=xxx,
        ...),
    val=dict(
        type=dataset_type,
        arg1=xxx,
        ...),
    test=dict(
        type=dataset_type,
        arg1=xxx,
        ...),
```

#### 



