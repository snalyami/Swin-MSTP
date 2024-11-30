
# Swin-MSTP

This is the code of the paper: [Swin-MSTP: Swin Transformer with Multi-Scale Temporal Perception for Continuous Sign Language Recognition]()

The code will be added soon ...
## Proposed Swin-MSTP Framework

<img src="https://lh3.googleusercontent.com/d/1Vwcv5uLiG_76Dt1GvijFqjN6YnP389H5" />

## GradCam Visualization

![hippo](https://drive.google.com/thumbnail?id=1cwEqqM2iy0C1_E_WasPwNXZDbCRTttX8)

### Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. Modify the dataset_preprocess.py file to reflect the location of the dataset.

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### Training
To train the Swin_MSTP model on phoenix14, run the command below:

`python main.py`

The Swin-Small architeture is used by default. If you would like to train your model using the Swin-Tiny strcuture, in the baseline.yaml file, change the c2d_type to 'swin_t'.  

### Inference
To evaluate the trained model, run the command below：

`python main.py --load-weights {name}.pt --phase test`

### Results

| Model                | WER on  Dev | WER on Test |                       Pretrained model                       |
| :---------------------- | :--------: | :---------: | :----------------------------------------------------------: |
|  Swin-MSTP<sub>tiny</sub> |    18.9    |    19.0     | [[GoogleDrive]](https://drive.google.com/file/d/1TtN5bam3mA52PfXxh5BxhXihWqJg-vmp/view?usp=sharing)|
|  Swin-MSTP<sub>small</sub>  |    18.1    |    18.7     |  [[GoogleDrive]](https://drive.google.com/file/d/1cK-h0Z8HsSlfEeqkgBruVYVvjccU8c_x/view?usp=sharing) |

The framewoek has also been trained on Phoenix2014-T, CSL, and CSL-Daily. The models are avaibale upon request.

### Acknowledgment

The code is based on [VAC](https://github.com/ycmin95/VAC_CSLR). We thank them for their work!
