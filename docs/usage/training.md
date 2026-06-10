# Training

## YOLO regression (baseline)

```bash
uv run python scripts/train.py --model yolo11
```

Config lives in `configs/models/yolo11.yaml`. Change `model:` to swap variant (nano/small/medium/large/xlarge).

## Custom PyTorch CNN

Training via the `RegressionTrainer` class in `src/cattle_weight_regression/training/trainer.py`.

```python
from cattle_weight_regression.training.trainer import RegressionTrainer
from cattle_weight_regression.models.pytorch.cnn import CattleWeightCNN

model = CattleWeightCNN(backbone="resnet50")
trainer = RegressionTrainer(model, train_loader, val_loader)
trainer.train(epochs=50, output_dir=Path("outputs/models/resnet50_run"))
```

All runs are automatically logged to MLflow.

## Efficient use of GPU/memory

Was originally having troubles with feeling like training was a lot slower than expected. I assumed it wasn't using GPU, turns out it was. The issue was just slow data loading. This has been improved to ~4x training speed by setting 4 workers for data loading, with `persistant_workers=True` in the PyTorch `DataLoader`.