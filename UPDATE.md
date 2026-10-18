# Update an existing installation

1. Stop the running local engine.
2. Extract the update ZIP into a temporary folder. Copy all files and folders directly inside it into your existing project folder, replacing the source files. The update ZIP has no additional enclosing project folder.
3. Keep your existing `local/` and `.venv/` folders and your icon/launcher files. This ZIP does not contain or replace your private data.
4. Use `start_web_windows.bat` for normal mode or `start_training_windows.bat` for the developer laboratory.

Your existing detector images, labels, selected detector, zones, and visit records remain available under the same profile name. The original icon supplied in `docs/icon.svg` is unchanged. A separately customized EXE or shortcut was not included in the uploaded source; retain it. If it embeds the application source, rebuild it to include these changes.

Activity recognition requires a developer-trained and evaluated model. No eating/drinking-trained model is bundled in this source update. See `docs/BEHAVIOR.md` to prepare a shared model that end users can use without labeling or training.
