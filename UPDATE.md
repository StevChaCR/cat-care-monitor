# Update an existing installation

Version 0.4.3 adds **Overview → Reset reports** in normal and developer mode. Disconnect the camera, press the button, and confirm the current profile. It clears all saved station visits, eating/drinking records, durations, and monitoring coverage for that profile and replaces the CSV exports with empty reports. Training images, clips, labels, models, zones, other profiles, and historical training metrics are retained. If a CSV is locked by Excel, close it and try again; the database is not reset until both existing exports can be moved safely.

Version 0.4.2 adds image trash and restore in **Cat library**, with permanent emptying of explicitly selected trashed images and their labels. It adds a **Behavior lab → Dataset checklist** with actual counts, proposed split membership, missing activity/station examples, unreviewed clips, and overlapping recording periods. When the first random split lacks coverage, preparation now searches for a valid assignment of whole periods before rejecting the data. Training, validation, and test remain separate; no per-class minimum has been lowered.

Version 0.4.1 adds **Food only / Water only / Both** behavior collection, station filters, recoverable clip trash, restore, batch cleanup of Cannot tell clips, and confirmed permanent emptying of trash. Choose the station before starting collection; clips still require activity review. See `docs/BEHAVIOR.md` for cleanup and focused collection instructions.

Version 0.4.0 keeps one selected detector, continues training from its weights, preserves recording-session split assignments, compares the previous cat-only detector against the candidate on the same test images, and bundles the selected detector for new installations. The performance controls from 0.3.1 remain available. See `docs/SHARED_MODEL.md`.

1. Stop the running local engine.
2. Extract the update ZIP into a temporary folder. Copy all files and folders directly inside it into your existing project folder, replacing the source files. The update ZIP has no additional enclosing project folder.
3. Keep your existing `local/` and `.venv/` folders and your icon/launcher files. This ZIP does not contain or replace your private data.
4. Use `start_web_windows.bat` for normal mode or `start_training_windows.bat` for the developer laboratory.

Your existing detector images, labels, selected detector, zones, and visit records remain available under the same profile name. The original icon supplied in `docs/icon.svg` is unchanged. A separately customized EXE or shortcut was not included in the uploaded source; retain it. If it embeds the application source, rebuild it to include these changes.

Activity recognition requires a developer-trained and evaluated model. No eating/drinking-trained model is bundled in this source update. See `docs/BEHAVIOR.md` to prepare a shared model that end users can use without labeling or training.
