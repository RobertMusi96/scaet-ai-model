# SCAET-AI Model

Multimodal classifier for distinguishing non-metallic inclusions from artifacts in automated SEM/EDS analysis of steel samples.

## Description

This model takes as input a backscattered electron (BSE) image patch and the  quantified energy-dispersive X-ray spectroscopy (EDS) 
composition of a feature detected by automated SEM/EDS analysis, and predicts the probability that the feature is a true non-metallic 
inclusion (NMI). It is intended as an extension of compositional filtering in steel cleanness analysis.

## Performance

Evaluated on a held-out test set of 618 features (see dataset 10.5281/zenodo.20758431):

- AUC-ROC: 0.99
- F1 score (false NMI class): 0.93
- Precision (false NMI class): 0.92
- Recall (false NMI class): 0.94
- Classification threshold: 0.88

## Repository contents

- `models.py` — model architecture (multimodal classifier with EfficientNet-B0 image branch and MLP EDS branch)
- `preprocessing.py` — preprocessing function for the public webdataset format
- `inference_example.py` — load the model and predict on a sample from the public dataset
- `inference_own_data.py` — load the model and predict on a user-provided BSE image and EDS composition
- `checkpoints/best_model.pt` — trained model checkpoint
- `requirements.txt` — Python dependencies

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run inference on the public dataset

Download the dataset from Zenodo ([10.5281/zenodo.20758431]) and place the tar files and metadata in a `data/` folder, then:

```bash
python inference_example.py
```

### Run inference on your own data

Edit the paths and EDS composition in `inference_own_data.py`, then:

```bash
python inference_own_data.py
```

## Preparing your own data

To apply the model to BSE images and EDS measurements from your own SEM/EDS analysis, the input must be prepared to match the format the 
model was trained on.

**BSE image patch**

- Extract the local patch around the detected feature using its bounding box
- Add a margin of approximately 5 µm (15 pixels at 0.33 µm/pixel) around the bounding box to include morphological context
- Save the patch as a 16-bit grayscale PNG file
- No further normalization is needed — the preprocessing script handles scaling and centering

**EDS composition**

- Provide the quantified composition as a Python dictionary mapping element symbol to weight percentage (e.g. `{"Fe": 53.9, "Al": 22.2, "O": 18.4}`)
- Elements not present in your data can be omitted — they are treated as 0 wt%
- The preprocessing script standardizes the composition using the training-set statistics from `metadata_public.json`

## Scope and limitations

The model was trained on features from polished Ti-ULC steel samples using the following acquisition parameters:

- Field emission SEM (JEOL JSM-7200F) with silicon drift detector (Oxford Instruments Ultim Max 100)
- Acceleration voltage: 15 kV
- Magnification: 400×
- Image resolution: 0.33 µm/pixel
- EDS analysis time: 1 to 3 s per feature
- Minimum feature size: 2 µm equivalent circle diameter

Users with substantially different steel grades, acquisition parameters, or feature size ranges should expect different performance. The 
model is intended as a complementary classification step on top of compositional filtering, not as a standalone replacement for expert 
inspection.

## Related resources

- **Publication**: 10.1007/s00501-026-01752-3
- **Dataset**: 10.5281/zenodo.20758431


## License

The code in this repository is released under the MIT License (see LICENSE).

The model weights in `checkpoints/best_model.pt` are released under the Creative Commons Attribution 4.0 International License (CC-BY-4.0), 
see MODEL_LICENSE.md.

## Contact

For questions or feedback, please open an issue on this repository or contact robert.musi@unileoben.ac.at.