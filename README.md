# VDN: Variant-Depth Network for Efficient Motion Deblurring via Scale Invariance
by Cai Guo, Qian Wang, [Hong-Ning Dai](https://www.henrylab.net), [Ping Li](https://web.comp.polyu.edu.hk/pinli/).

Pytorch Implementation of CASA2022 "VDN: Variant-Depth Network for Efficient Motion Deblurring via Scale Invariance"
![Pipeline of VDN](./docs/VDN.png)

### Results on the GoPro evaluation dataset
![Pipeline of Results1](./docs/Results1.png)

### Results on the HIDE evaluation dataset
![Pipeline of Results2](./docs/Results2.png)

## Dependencies
python
```
conda create -n vdn python=3.7
conda activate vdn
```
pytorch
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

## Testing
Please copy test samples into './test_samples'. Then running the following command.
```
python test.py
```
