# Release model resources

After evaluating a real activity model, use `tools/bundle_behavior_model.py` to create `behavior/model.json`, `behavior/weights.npz`, and `behavior/encoder.pth` here. Normal mode automatically loads them. No real eating/drinking-trained model is included in this source update.

Keep training images, clips, datasets, private checkpoints, and camera credentials outside this directory. Include only the explicitly reviewed model resources in a release.
