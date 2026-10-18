# Train and distribute the shared WatchMyCat detector

The product has one selected cat detector. New users receive the developer's included detector and connect their own local RTSP camera; they do not have to collect images, choose YOLO variants, or train. The developer laboratory remains available through `start_training_windows.bat`.

This source update prepares the workflow. It does **not** contain Steven's private trained weights or the 98 reviewed Mila images, and it does not claim that a new model has already been trained. Those files are on the developer's Windows computer. Existing custom weights remain selected after installing this update.

## Continue from the current model

1. Copy this update over the existing project, keeping `local/`, `.venv/`, and customized launcher/icon resources.
2. Run `start_training_windows.bat` and open the same developer profile used for the current detector. The camera form now shows only the current detector, without alternate YOLO choices.
3. Add and review examples in **Cat library**. Keep the shared training corpus in one developer profile. Each session represents one genuine recording period, not a renamed copy or neighboring frames assigned to different splits.
4. Disconnect the camera. Keep **GPU 0** selected on the development computer. In **Train detection**, choose a maximum epoch count and press **Train current detector**. The existing default is 60 with early stopping; additional epochs alone do not guarantee improvement.
5. Each run begins from the selected checkpoint and saves a separate candidate. The current weights are not overwritten. The report includes precision, recall, mAP50, and mAP50–95. For a cat-only starting model, it evaluates the previous detector and the candidate against the same test images and displays their differences.
6. Inspect errors and complete recordings, then choose **Use this candidate locally** only when the change is useful. Check missed cats, false cats, broken track identities, duplicate visits, and visit duration. A completed training run is not proof that the candidate is better.

The first developer training run on a fresh installation can use a starter detector internally. Once a cat-only detector has been trained, subsequent runs continue from that selected detector. The starter models are not alternatives in the end-user interface.

## Stable evaluation data

Training writes `local/web/<profile>/detector_split.json`. Existing training, validation, and test recording periods keep their roles across subsequent runs. When possible, the first run after this update inherits the original split manifest associated with the selected checkpoint. New periods go into training instead of silently moving old test footage into training.

Retain this file, the session identifiers, and the corresponding original periods. Deleting or renaming periods does not create independent data. If a held-out split becomes empty or lacks a labeled cat, training stops instead of silently reallocating it. Exact copies appearing in different periods are rejected.

This grouping is by recording period, not an automatic guarantee of independence by animal, house, or camera. Adjacent recordings with different names can still overlap. For a product-wide claim, reserve a separate evaluation corpus containing genuinely unseen cats/cameras and keep it out of all development training and tuning. New training data added after the first split is not automatically an unseen-cat benchmark. Do not repeatedly choose models against the same test set and call it an untouched final test.

## Improve beyond Mila

Keep Mila as a regression case, and add consented or appropriately licensed examples covering:

- Different cats: coat colors and lengths, body sizes, ages, and proportions.
- Different homes and cameras: angles, mounting distances, backgrounds, resolutions, and compression.
- Daylight, indoor lighting, infrared, shadows, and reflections.
- Walking, entering/leaving the frame, crouching, jumping, partial occlusion, bowls, and litter-box edges.
- Reviewed negative scenes: dogs and other pets, blankets, toys, furniture, empty bowls, and empty rooms.

Review every visible cat box and preserve correctly labeled empty scenes. Prefer varied failure cases to many nearly identical frames. There is no number of Mila-only images or epochs that establishes reliability for all users.

## Include the selected detector

After evaluating the candidate and selecting it locally, disconnect the camera and press **Include current detector in release**. Alternatively:

```powershell
.\.venv\Scripts\python.exe tools\bundle_detector_model.py
```

The command reads the currently selected checkpoint. A specific evaluated checkpoint can be provided with `--weights`.

The generated resources are:

```text
assets/detector/model.pt
assets/detector/model.json
assets/detector/MODEL_CARD.md
```

The tool checks cat-only detection classes and a held-out evaluation report, removes training checkpoint metadata and local training paths, reloads the saved checkpoint, and writes a checksum and aggregate metrics. New reports tie their evaluation to the exact original checkpoint hash. Existing legacy reports without a hash are accepted as a migration path; re-evaluate those checkpoints before making a public accuracy claim.

The exported tensors use Ultralytics' normal checkpoint serialization. Only load release weights from a source you trust. A checksum detects corruption; it does not authenticate the publisher. No training photos, clips, private profile settings, visit database, or camera credentials are copied by this bundler.

This step creates local release resources. It does not publish anything to GitHub. It also does not certify performance on other cats or cameras; the model card states the evaluation scope explicitly.

## End-user startup

Include the generated `assets/detector/` directory alongside the application source or executable resources. New installations automatically select it. Existing installations still using a starter model also prefer the included detector. Existing developer profiles with a custom checkpoint keep that checkpoint.

Use `start_web_windows.bat` for the normal end-user interface. Users connect an authenticated local RTSP camera, mark their stations, and follow observations. The included detector loads locally without downloading a generic detector. No model selector or training workflow is shown in normal mode.

An executable embedding Python must be rebuilt with these assets; copying source files does not change an already frozen binary. Source releases remain usable from GitHub with the provided launchers. GitHub Pages hosts the interface; RTSP capture and inference still run on the user's computer.

## Detection and behavior are distinct capabilities

The shared detector learns where cats are. A label containing only `cat` does not teach eating, drinking, or elimination. The existing **Behavior lab** trains a separate temporal activity component using reviewed food/water clips and includes it using `tools/bundle_behavior_model.py`. Both components can be shipped in one app without asking end users to train either one.

Current visit counts measure station presence. The activity component can estimate probable eating and drinking only after real activity training and evaluation. The current activity labels do not include urination or defecation. Do not market station presence as confirmed elimination or the detector packaging step as completed behavior training.

## GitHub preparation

The project retains its existing AGPL-3.0-only license and upstream notices. The `.gitignore` permits only the explicit generated release model paths while continuing to exclude private checkpoints and `local/`. The public-file checker verifies release checksums before accepting those paths.

After generating the resources, review what will be published:

```powershell
git add .
.\.venv\Scripts\python.exe tools\check_public_files.py
git status --short
```

Publish the selected release assets and source. Retain private images, annotations, other candidate weights, camera URLs, profiles, and databases locally. Keep the previous working training checkpoint as a rollback copy. The bundler does not delete it.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/), [validation](https://docs.ultralytics.com/modes/val/), [deployment practices](https://docs.ultralytics.com/guides/model-deployment-practices/), and [licensing](https://www.ultralytics.com/license).
