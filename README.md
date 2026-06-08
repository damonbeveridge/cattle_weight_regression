# Cattle Weight Estimation

This project takes a [dataset of cattle images](https://github.com/bhuiyanmobasshir94/CID) and produce a deep learning model to estimate the weight of unseen cows.

Due to the design of the dataset, we can approach two architecture types:
- Standard 1 input image to 1 output values.
- 4 input images of standardised orientations to 1 output value, giving the model more information about the 3D shape of the cow.

## Project architecture

**Choice:** Since we have no bounding box labels, jumping straight using PyTorch CNN approach using a ResNet or EfficientNet backbone. It is simpler to implement correctly and produces the same class of model. YOLO is difficult, but worth attempting the following after getting baseline above working.

YOLO classify as a whole-image feature extractor (YOLO interchangeable with deep learning model):

```
Image → YOLO classify backbone → feature vector → Linear regression head → weight (kg)
```

Use `task: classify` and change the classification head to output 1 value instead of N classes. Don't need bbox labels — just the images and weights. This is the fastest path to a first result and is effectively the same as using YOLO's backbone as a pretrained encoder.

Update `configs/models/yolo11.yaml` to:

```
task: classify
```

The catch is that Ultralytics classify expects images sorted into class subdirectories (`train/class_a/`, `train/class_b/`, etc.), which doesn't match regression. You'd need to either:

- Use the YOLO backbone purely as a feature extractor in PyTorch (extract embeddings, then train a regression head) — this is cleaner and is what the `CattleWeightCNN` pattern in `pytorch/cnn.py` does with a ResNet backbone.
- Or use Ultralytics with a custom trainer that overrides the loss to MSE.

## Future improvements/ideas

**YOLO segmentation pipeline**

Find a dataset containing bounding box labels so that I can build a YOLO pipeline:

```
Image → YOLO detect (localise the cow) → Crop → Regression CNN → weight (kg)
```

Stage 1: Train YOLO `detect` to draw a bounding box around each cow. This uses YOLO in its native task and can get the full benefit of its pretrained backbone.

Stage 2: Crop the detected region from each image, then feed the crop into a separate regression model — either `CattleWeightCNN` from `pytorch/cnn.py` or a simple scikit-learn regressor on extracted features.

**Why this is good:** Each stage is independently interpretable and testable. Can evaluate detection quality separately from regression quality. It's also the most robust to background clutter.

**Downside:** Need bounding box annotations (x, y, width, height) in labels CSV or as YOLO-format `.txt` files alongside the images. Current dataset doesn't have them, so we'd need to label them (tools: Label Studio, Roboflow).

YOLO expects label files in this format — one `.txt` per image, same stem name:

```
# class_id  cx  cy  width  height   (all normalised 0-1)
0  0.512  0.498  0.843  0.761
```

**YOLO pose + body measurements pipeline**

Annotate keypoints on each cow (hip joints, shoulder, nose, tail root — 5–8 points). YOLO pose estimates these points. Can then derive biologically meaningful measurements (body length, hip width, heart girth proxy) and feed them into a linear or gradient-boosted regressor.

This is the most defensible scientifically and produces a model whose inputs a vet could reason about. It also needs the most labelling effort. To be considered for a later experiment once you have baselines from the segmentation pipeline.

## Documentation



## License

