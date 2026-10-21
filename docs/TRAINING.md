# Improve the cat detector

Train a detector to recognize **cats** first. Station geometry and the visit state machine produce the visit records. Recognition of actual eating, drinking, or elimination would require a separate validated behavior or pose model.

## 1. Record a baseline

Run preview and record where the generic model misses cats or creates false detections. Include passes near bowls that should not count as visits. Compare the same held-out clips and camera settings before and after fine-tuning.

```powershell
.\.venv\Scripts\python.exe app.py --preview
```

Press S to collect complete original-resolution camera frames without display overlays. Each application run creates a recording-session folder under `local/captures/`. Capture cat-positive and empty scenes in each session. Use separate times/days and include the difficult infrared/occluded cases. A small initial pilot is useful; image diversity and correct labels matter more than a fixed image count.

For a detector intended for many users, gather varied cats, coats, camera placements, backgrounds, and cameras with permission. Do not judge that general model solely on Mila's room.

### Infrared cameras and missed cats

A generic color-photo detector may miss a cat in infrared footage even when the animal is obvious to a person. A larger model, lower confidence, or contrast enhancement is not a demonstrated fix until checked on your camera's unseen footage and empty scenes.

Use the images saved with **S**, rather than screenshots containing station rectangles or detection labels. Screenshots can be small, cropped, and visually altered; they are a different input from the original camera feed. If a focus crop cuts off the cat, uncheck **Focus detection around the selected stations** while comparing the full frame.

Collect difficult infrared poses, sitting and curled-up cats, partial occlusion, different distances, and cat-free scenes with the same bowls, furniture, and glowing reflections. Keep several times/days as independent sessions; three neighboring images are one scene, not three independent sessions. Manually label cats the model missed. Hold out complete infrared sessions and compare missed visits as well as false detections after training.

The labeling window also fits the full camera image to the screen, while saving labels in normalized original-image coordinates.

## 2. Label manually

```powershell
.\.venv\Scripts\python.exe tools/label_frames.py
```

- B: draw a box around every visible cat; Enter confirms the box.
- N: explicitly confirm an image contains no cat.
- S: save the reviewed label and move to the next image.
- Q: leave unfinished images unlabeled.

Use one class, **cat**, with class ID 0. Draw consistent boxes around the visible extent of the cat. Do not label a missed cat as an empty scene. Automatic predictions are not treated as ground truth. Existing reviewed images are skipped unless `--review-existing` is supplied.

## 3. Split by recording session

```powershell
.\.venv\Scripts\python.exe tools/prepare_dataset.py
```

At least three distinct sessions are needed. The split is approximately 70/15/15 by session, with at least one validation and one test session. With only three sessions, each split gets one. Every image needs an explicit label file, including empty negative files. Each split must include labeled cats.

Neighboring frames from the same recording stay together; exact image copies across sessions are rejected. Use genuinely different sessions, not copies of the same clip under different folder names. For a broad public model, also hold out cats/cameras or homes and report their results separately.

The helper writes `local/dataset/data.yaml` and a split manifest. An existing dataset is preserved; choose `--output local/dataset-v2` for a new preparation.

## 4. Fine-tune locally

If this virtual environment has compatible PyTorch CUDA installed:

```powershell
.\.venv\Scripts\python.exe tools/train_detector.py --device 0 --epochs 60 --imgsz 640 --batch 2
```

For CPU, use `--device cpu`; a short pilot with `--epochs 5` checks the workflow but is not a quality target. The default pretrained model is the smaller `yolo26n.pt`. Try `--model yolo26s.pt` and/or `--imgsz 960` when memory and throughput permit. Batch 2 is a starting point, not a memory guarantee.

The script uses early stopping, then evaluates `best.pt` on the held-out **test** sessions and saves `held_out_metrics.json`. This is actual evaluation output produced only when training is run; the repository does not contain fabricated performance results.

Keep the test sessions out of threshold tuning. Choose operating confidence on validation sessions. Favor recall for missed cats while checking false positives in empty scenes and false visits per hour.

## 5. Compare the visit measurements

Use Browse in the connection dialog to load the resulting `best.pt`. Class ID may change from COCO cat 15 to custom cat 0; the application resolves the class by its name.

Compare both models on the same held-out clips or manually observed visits:

| Check | What to record |
|---|---|
| Missed visits | Real station entries not logged |
| Extra visits | Passes, objects, or repeated counting of one visit |
| Duration error | Difference from manually observed first/last presence |
| Interrupted coverage | Time with unavailable video or slow inference |
| Domain changes | Day versus infrared, unseen cats and camera positions |

Better mAP alone does not prove better station counting. Keep the old model and settings until the new version improves the actual visit measurements. Generic detection and individual cat identity are separate tasks.

## Private training data

Captures, labels, datasets, checkpoints, and metrics are under `local/` and excluded from Git. Keep private images out of the public source archive. The application does not upload them automatically. Publishing a custom model is a separate release decision; respect image permissions and upstream model licensing.

## Remove or correct training images

In **Cat library**, use **Move to trash** on an image card or inside its box editor. Its JPEG, metadata, and reviewed label move together. Trashed images are excluded from future dataset preparation. Use **Show → Trash → Restore image** to recover the original image, label, and session. A confirmed empty-scene label remains distinct from an image that was never reviewed.

To reclaim disk space, use **Empty image trash** and confirm the displayed count. Only the explicitly listed trash items are permanently erased, up to 600 per action. Active images, model weights, visit records, behavior clips, and historical training runs are separate. Historical training runs retain the dataset snapshots they prepared earlier.

For a useful photo with an incorrect label, open it, press **Clear boxes**, and draw one box per actual cat. A photo of Mila alone should contain one cat box. Keep useful empty scenes and correctly annotated examples rather than deleting every difficult photo. Images with an unreadable preview can still be moved to trash. Image and label changes are blocked while training is running.

Deleting an example does not remove its influence from weights that were already trained on it. Clean the labels and train a new candidate. When investigating a regression, compare with a previously working checkpoint and use that checkpoint as a base when appropriate; this update does not automatically replace or erase the currently selected weights.
