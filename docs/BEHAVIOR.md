# Eating and drinking recognition

## Product flow

End users connect their local RTSP camera, mark the stations, and leave the local engine running. The normal launcher does not expose the image library, behavior labeling, epochs, or training. Its server also rejects the training API routes. No images are collected for training in normal mode.

The developer builds and evaluates the shared detector and activity model before distribution. Users are not required to create a personal dataset. A developer model trained only on Mila remains a camera-specific prototype until it is evaluated on other cats, rooms, lighting conditions, and cameras.

This source update includes the capture, labeling, training, bundling, inference, and event pipeline. **It does not include a trained eating/drinking model.** Without one, the app continues to record station visits and explicitly shows activity recognition as unavailable. It never converts presence near a bowl into an eating or drinking event.

## Two launchers

- `start_web_windows.bat`: normal user mode.
- `start_training_windows.bat`: developer laboratory, including the existing detection-photo workflow and the new **Behavior lab**.
- Linux/macOS: `python web_app.py --open` or `python web_app.py --open --training`.

The laboratory flag is a product separation, not an account-authentication system. The server remains bound to loopback and requires its per-process token. Files remain accessible to the operating-system user that owns them.

## Developer collection and review

1. Launch the training mode and keep your existing cat profile. Existing photos, reviewed boxes, chosen detector, zones, and visit database remain in `local/`.
2. Fix the camera position. Configure food and water zones with enough space to see the cat's head and the rim of the bowl. Keep the stations separate. A hidden or tiny head cannot reliably establish an action.
3. Open **Behavior lab** and create a session for one real recording period, for example `Camera A - Monday evening`. Connect the camera and enable clip collection. This is explicitly opt-in.
4. Choose **Collection area → Water only**, **Food only**, or **Both configured stations**, then press **Collect behavior clips**. Stop collection before switching areas. Reuse the same session when switching areas during the same real recording period. Each stored sequence contains 16 JPEG frames sampled near 4 FPS, cropped around the selected station with context. Stations are sampled independently of YOLO, so missed cats, resting animals, and empty scenes can be reviewed too. No audio or RTSP credentials are stored. Collection is limited to 120 clips per camera run and 600 active clips per profile.
5. Review each clip as a sequence. Use **E** for eating, **D** for drinking, **O** for other activity, or **X** for cannot tell. Other must include sniffing, sitting, lying beside the bowl, grooming, walking past, and empty scenes. Exclude occlusions, multiple cats, transitions between activities, or ambiguous clips rather than guessing a label.
6. Use independent real periods for training, validation, and testing. A new session name or a reconnection does not make adjacent footage independent. The helper rejects overlapping source time ranges and duplicate clips. Each split must contain all three activities and Other examples from both stations. At least three examples of each class are required in training and one per class in validation and test; these are software minimums, not an adequate-data guarantee. An individual period can contain a single activity; the split planner can combine multiple complete periods to meet coverage. It keeps the initial session assignment when valid and searches for a class-balanced whole-period assignment when needed. It never splits one period between sets to meet a count.

## Why three periods, and why 14 / 7 / 7 may still be blocked

A **recording period** is one genuine recording occasion, for example Monday breakfast, Tuesday evening, and Wednesday morning. Occasions on the same day can also be distinct. There is no universal waiting interval that guarantees independence, and three different days are not mandatory. Keep neighboring frames, one continuous event, and simultaneous food/water recordings in the same session. Empty session names, copied clips, or reconnecting during the same event do not create new independent evidence.

This program requires at least three usable periods because it reserves separate groups for **training**, **validation** (model selection), and **test** (held-out evaluation). It is an application rule, not a requirement imposed by YOLO. Mixing nearly identical neighboring frames across these groups can overstate performance. The grouping remains by recording occasion, not an automatic guarantee of generalization across cats or homes.

The current software minimums are:

| Set | Eating | Drinking | Other | Other station coverage |
| --- | --- | --- | --- | --- |
| Training | 3 | 3 | 3 | At least one food and one water clip |
| Validation | 1 | 1 | 2 | At least one food and one water clip |
| Test | 1 | 1 | 2 | At least one food and one water clip |

Validation and test each need two Other clips because one clip belongs to a single station. Thus **14 Eating, 7 Drinking, and 7 Other** satisfy the global numeric minimums, while training can still be blocked if the clips are all in one period, the seven Other clips cover only water, some clips remain unreviewed, periods overlap, or complete periods cannot be assigned to cover every set. These are the program's preparation thresholds, not an accuracy guarantee.

Open **Behavior lab → Dataset checklist** and press **Check dataset**. The panel shows total class counts, unreviewed clips, usable periods, proposed membership, missing counts, and Other coverage at food/water in each set. Expand **What are recording periods?** to see session names and captured time ranges. Empty sessions do not count as usable periods. Labels are counted from the reviewed activity, never inferred from the collection area.

When the initial random partition fails but a valid whole-period partition exists, preparation now tries to find that partition before rejecting the data. The search is bounded to keep the lab responsive. The proposed partition can change as active clips and periods change; reserve a separate independent final-release evaluation corpus outside development tuning. Frame integrity and exact duplicates are checked again when preparing the real training data.

Stop clip collection and **disconnect the camera** before training; a connected camera also disables the training button. If dataset preparation passes but the training process fails, its progress message and private `training.log` identify a separate dependency, encoder-download, or memory problem.

Reference: [scikit-learn on grouped and time-dependent evaluation](https://scikit-learn.org/stable/modules/cross_validation.html).

## Collect a specific station

To focus on drinking examples, keep the camera connected, choose the real recording period, select **Water only**, and start collection. Stop when the useful moment ends. For food, repeat with **Food only**. Leave both station zones configured; the selection changes only which clips are saved, while monitoring continues for both stations.

The selector chooses a camera area, not a known action. A water crop may show drinking, sniffing, resting, or an empty bowl. Clips remain unreviewed until you label them **Drinking**, **Eating**, **Other**, or **Cannot tell**. Keep useful Other examples at both stations; a model trained only on positive examples cannot reliably distinguish a visit from the activity. Collection begins a fresh sequence after each start or area change, and never labels clips from their station alone.

## Remove bad recordings

1. In **Review behavior clips**, filter by activity and **Station**. Choose **All clips** to include previously reviewed clips. Use **Show 60 more clips** to reach older recordings.
2. Click **Move to trash** on a card or inside its review dialog. The clip leaves the active library and future training datasets, while its original frames, label, and recording period remain recoverable. Even a clip with an unreadable preview can be moved to trash.
3. To clean up rejected clips in a batch, choose **Cannot tell** and the desired station, then click **Move Cannot tell to trash**. This moves only rejected clips matching the current filters.
4. Choose **Show → Trash** to review removed clips and use **Restore clip**. Restoring preserves the original label; a restored Cannot tell clip remains excluded until you review it again.
5. To reclaim disk space, click **Empty trash** and confirm the displayed count. This permanently deletes up to 600 explicitly listed trash clips per action and cannot be undone. Clips added to trash after that selection are not included.

**Cannot tell** alone excludes a clip from training but keeps its files. Moving to trash is recoverable and does not free disk space until the trash is emptied. Trash lives under `local/web/<profile>/behavior/sessions/trash/` and is ignored by dataset preparation. Active clip capacity excludes trash. Existing trained weights and visit/activity records are not changed by cleanup; removing source clips affects future dataset preparation. Clip changes are blocked while a training job runs.

For a shared product, collect consented examples from multiple cats and cameras in the developer workspace and label their separate recording periods clearly. Hold out entire cameras/cats as well as days when assessing generalization. The current automatic split keeps recording periods together; it does **not** automatically certify independence by animal or camera identity. Do not present a random frame split or same-camera result as proof of broad compatibility.

## Inference performance

In **Camera & zones**, **Use faster AI · 640 px** applies a 640-pixel inference input immediately, including while the camera is connected. It retains the selected custom detector. The complete preview and station clip crops retain their framing. Existing installations keep their selected input size until this button or the setup form changes it.

The panel shows the actual processor, input size, processing milliseconds per frame, and latest prediction age. A 640 input processes substantially fewer pixels than 960 or 1920, but small or distant cats may become harder to detect. Compare real missed detections before keeping the setting. There is no guaranteed FPS improvement on an untested computer. Training on 98 images does not require running inference at the training image size.

The detector transfers its box table from the GPU once per inference instead of requesting each field separately. Camera capture still keeps only the latest frame and detection remains asynchronous. Fractional measured frame rates below 0.5 FPS are now displayed rather than rounding the sample history down to zero. This release does not force FP16 on older GPUs, install CUDA, export TensorRT, or replace a custom detector with a generic model. If CPU is displayed, GPU acceleration is not being used; inspect the existing diagnostics before changing dependencies.

Inference settings: [Ultralytics prediction arguments](https://docs.ultralytics.com/modes/predict/).

## Local training

Disconnect the camera and stop detector training before starting activity training. The activity learner uses frozen official MobileNetV3-Small visual features and a small learned temporal convolution over 16 frames. The official encoder is about 10 MB and is downloaded once during developer training. It is not itself a pretrained cat-action recognizer.

Training reports real preparation counts and epoch progress. Early stopping uses validation macro F1, with validation loss as a tie-breaker. Evaluation uses held-out test clips and reports per-class precision, recall, F1, example counts, and a confusion matrix. The app does not promise better behavior recognition because a run completed.

Use **Use this candidate locally** to test the selected activity model without replacing the cat detector. Low-confidence or conflicting decisions remain unknown. Test complete recordings not used for training: mark the true start and end of every feeding bout and compare false sessions, missed sessions, duplicates, and duration errors. Clip accuracy is not event-level accuracy.

## Include the model for end users

After training, inspect `local/web/<profile>/behavior/training/<run>/progress.json` and evaluate the model on complete unseen recordings. Then run:

```powershell
.\.venv\Scripts\python.exe tools\bundle_behavior_model.py --run "local\web\PROFILE_ID\behavior\training\RUN_ID"
```

Replace `PROFILE_ID` and `RUN_ID` with the actual folder identifiers. The tool creates `assets/behavior/model.json`, `weights.npz`, and `encoder.pth`. Include these three files in the release or executable resources. They contain the model and its metrics, not the original training clips. Models trained on private data can still disclose information; do not publish private-data-derived weights without considering that data's permission and privacy requirements.

Normal mode automatically uses the included activity model. It reads the encoder from the package, so the end user does not have to label clips, train, or download this encoder. A profile previously used for local candidate testing can select **Use included model** in developer mode to return to the release model.

The custom head is loaded from checked numeric NumPy arrays with `allow_pickle=False`; no custom executable Python checkpoint is accepted by this activity loader. The official encoder is checked against its full SHA-256 and loaded as tensor weights. Checksums detect corruption; release provenance and software permissions still matter.

If your EXE is only a launcher, retain it and update the source it starts. If it embeds the Python source, rebuild that EXE and include the asset files. This update does not build a Windows binary or installer.

## Live decisions and sessions

Recognition runs in a separate worker so camera preview does not wait for activity classification. It processes short windows only when a fresh cat detection is near the station. A model's held tracking box does not count as fresh evidence. Multiple cats, missing/stale detections, sparse video, weak scores, and a prediction inconsistent with the station are unknown.

The initial conservative gate requires a model score of at least 0.85 and a margin of at least 0.15 over the next class. These scores are not calibrated probabilities. Two seconds of consecutive accepted action evidence are required to open an activity session. Six seconds of observed departure or reliable Other evidence close it. These are starting engineering settings requiring event-level evaluation.

Brief pauses keep the same session. Unknown periods pause the activity timer and flag the session incomplete. Reacquisition does not create another session without an observed end. Only intervals between consecutive accepted action observations within 2.5 seconds contribute to activity time. Offline time and lost detections are not filled in. On restart/reconnection, an unresolved session for the same model is resumed with an incomplete flag instead of silently creating a duplicate.

The dashboard separately shows **station visits** and **probable eating/drinking sessions**. Duration is an estimate from observed evidence; it does not measure grams, milliliters, urination, or defecation.

## Files and upgrades

- Existing detection photos and labels: `local/web/<profile>/sessions/`.
- Behavior clips and reviewed activity labels: `local/web/<profile>/behavior/sessions/`.
- Activity training runs: `local/web/<profile>/behavior/training/`.
- Visit and activity tables: `local/data/visits_<profile>.sqlite3`. New tables are created without deleting existing visits.
- Activity export: `local/reports/activities_<profile>.csv`.
- Shareable release model: `assets/behavior/` only after the developer bundles an evaluated candidate.

When upgrading, copy the new source into the same project and retain `local/`, `.venv/`, and your own icon/launcher resources. Existing detection photos are not moved or relabeled. Profile names still select separate workspaces; the button now says **Open profile**. Use the exact previous name to return to an existing workspace. Renaming a profile while retaining its data is a separate migration feature, not part of this update.

Source installations continue using their existing `local/` directory. A future frozen executable uses the account's application-data directory (`%LOCALAPPDATA%/WatchMyCat` on Windows), outside temporary executable extraction resources, so its records survive restarts. Existing source data is not automatically moved. This is persistent storage, not image encryption or a multi-user authentication system.

## Verification scope

The update has automated checks for session grouping, unknown-time handling, resumed session identity, station/action mismatches, dataset splits, capture timing, private APIs, normal-mode training rejection, numeric model integrity, and temporal-head training. The developer integration check uses the real official visual encoder with synthetic clips to exercise training, bundling, and inference; it does not establish accuracy on real cats. A live Eufy stream, native Windows packaging, full browser layout, GPU behavior, and real eating/drinking accuracy still require testing on the target system and real held-out recordings.

Technical references: [Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), [PyTorch serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html).
