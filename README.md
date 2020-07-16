# [ECCV 2020] Global and Local Enhancement Network For Paired and Unpaired Image Enhancement

<div align="center">
  <img width="90%" alt="GleNet" src="./paper/fig2.png">
</div>

A novel approach for paired and unpaired image enhancement is proposed in this work. First, we develop global enhancement network (GEN) and local enhancement network (LEN), which can faithfully enhance images. The proposed GEN performs the channel-wise intensity transforms that can be trained easier than the pixel-wise prediction. The proposed LEN refines GEN results based on spatial filtering. Second, we propose different training schemes for paired learning and unpaired learning to train GEN and LEN. Especially, we propose a two-stage training scheme based on generative adversarial networks for unpaired learning. Experimental results demonstrate that the proposed algorithm outperforms the state-of-the-arts in paired and unpaired image enhancement. Notably, the proposed unpaired image enhancement algorithm provides better results than recent state-of-the-art paired image enhancement algorithms.

<br/><br/>

## Enviroment setup
Our code is developed with TensorFlow v2. See requirements.txt for all prerequisites, and you can also install them using the following command.
```
pip install -r requirements.txt
```

## Paired Image Enhancement
Coming Soon...

## Unpaired Image Enhancement
Coming Soon...

## Cite
```
@inproceedings{fu2016,
  title = {Global and Local Enhancement Network For Paired and Unpaired Image Enhancement},
  author = {Kim, Han-Ul and Koh, Young Jun and Kim, Chang-Su},
  booktitle={European Conference on Computer Vision},
  year = {2020}
}
```
