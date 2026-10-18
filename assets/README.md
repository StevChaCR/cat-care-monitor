# Release model resources

After evaluating and selecting the shared cat detector, use **Include current detector in release** in the laboratory or run `tools/bundle_detector_model.py`. This creates `detector/model.pt`, `detector/model.json`, and `detector/MODEL_CARD.md`. New installations load that detector automatically. No trained detector is included in this source update because the private weights and 98 reviewed images were not part of the uploaded source.

After evaluating a real activity model, use `tools/bundle_behavior_model.py` to create `behavior/model.json`, `behavior/weights.npz`, and `behavior/encoder.pth` here. Normal mode automatically loads them. No real eating/drinking-trained model is included in this source update.

Keep training images, clips, datasets, private checkpoints, and camera credentials outside this directory. Include only the explicitly reviewed model resources in a release.
