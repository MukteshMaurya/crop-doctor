# Model asset: `unified_plant_resnet50.pth`

- **Source:** existing trained checkpoint from `crop doctor/unified_plant_resnet50.pth`
  (SHA256 `CBCD4DC5BF7D5B138A70F914C6C02721EC2FF43EF15761BEBF18374ADEA903A1`).
- **Architecture:** torchvision ResNet50, final FC layer replaced with
  `Linear(2048, 17)`.
- **Checkpoint format:** dict with keys `num_classes`, `class_names`,
  `model_state_dict`, `optimizer_state_dict` (the optimizer state accounts
  for the ~270 MB file size; only `model_state_dict` is used at inference).
- **Classes (17, in order):** apple, banana, blueberry, cherry, citrus, corn,
  grape, other_crops, peach, pepper_bell, potato, raspberry, soybean, squash,
  strawberry, tomato, wheat.
- **Note:** this checkpoint classifies crop/plant *type*, not disease.
- **Size:** ~270 MB — larger than GitHub's 100 MB file limit, so it must be
  tracked with Git LFS (see PROJECT_PROGRESS.md, Phase 5).
- The model is reused as-is: it is **not** retrained, pruned, or replaced.
