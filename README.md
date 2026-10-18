# Cat Care Monitor

## Version 0.4: one shared detector and developer training

Version 0.4.1 adds station-specific behavior collection and clip cleanup. In the Behavior lab, collect **Food only**, **Water only**, or both; review the actual activity before labeling. Move bad clips to recoverable trash, restore them from the Trash filter, or permanently empty the trash to reclaim disk space. See [the behavior workflow](docs/BEHAVIOR.md).

The normal web launcher now exposes camera setup and observations. Photo labeling and model training are developer tools, opened with **`start_training_windows.bat`** or `python web_app.py --open --training`. End users do not have to create a personal dataset.

The new behavior pipeline collects and reviews short station sequences, trains a separate temporal classifier, and can include that model in a release for automatic eating/drinking estimates. **No eating/drinking-trained weights are included in this source archive.** Until a developer trains, evaluates, and bundles a model, activity recognition is shown as unavailable and station-visit counting continues. Existing cat-detection training and private data are retained.

See **[Eating and drinking recognition](docs/BEHAVIOR.md)** for the developer workflow, release bundling command, event rules, upgrade instructions, and validation limits. The original desktop monitor remains available; the new behavior pipeline is integrated in the local browser engine.

A local Python application with browser and desktop interfaces. It connects to an existing authenticated **local RTSP camera** and estimates one cat's visits to a litter box, food station, and water station.

## Browser interface and GitHub Pages

On Windows, double-click **`start_web_windows.bat`**. It installs the dependencies in `.venv/` and opens **http://127.0.0.1:8765**. Keep the terminal open. Close the original desktop monitor first; use one monitoring instance per cat profile to avoid duplicate records. On Linux/macOS use `sh start_web.sh`, or run `python web_app.py --open` in an environment with the requirements installed.

The browser interface includes:

- Full-frame camera preview, persistent tracking boxes, and draggable station zones. Connect once for a preview, disconnect to draw on the last frame, save, then reconnect.
- In developer mode, a cat profile with its own photos, reviewed labels, station settings, and selected model. Profile names identify local workspaces, not individual animals in the video.
- In developer mode, JPEG/PNG import, a full-frame box editor, and explicit empty-scene labels.
- In developer mode, optional automatic camera capture independent of the detector, including missed cats and misclassifications. It keeps up to 120 images per camera run. Captures require review before training.
- In developer mode, a cancellable local training process with actual epoch progress and held-out test metrics. It requires reviewed data from at least three independent recording sessions. The new model is selected explicitly after review; training does not guarantee improved live counts.

Publish the **`docs/` folder** as a GitHub Pages site. The public site displays the interface and instructions for opening the local app. Camera credentials, photo import, and training are enabled only on the authenticated local engine. GitHub Pages does not run Python, open RTSP streams, or supply a training GPU. See [GitHub Pages setup](docs/GITHUB_PAGES.md).

The browser engine binds only to `127.0.0.1`. It uses a fresh session token and rejects other website origins; it does not expose a camera API to the internet. Open the app on the same computer that runs the engine. The public site and local app share the same frontend files; no cloud bridge or account is required.

Private web images and per-profile settings live under `local/web/<profile>/`. The existing desktop S captures remain under `local/captures/`. The two collections are separate. Private images and training checkpoints are excluded from the public archive; explicitly bundled release models can be included.

### Camera startup diagnostics (0.2.1)

Browser mode opens the camera before loading the detector. Video preview can appear while weights load. If model loading or detection fails, the camera stays available in **Preview only** mode; visit measurements pause until detection is restored. CUDA-related failures trigger one automatic retry on CPU.

The camera page shows the error stage, code, and redacted exception details, with a **Copy diagnostic** button. The latest safe diagnostic is also saved to `local/diagnostics/latest.json` and printed in the terminal. Camera URLs and their credentials are removed from these diagnostic messages. Do not paste your actual RTSP URL into an issue.

While the camera has not supplied a fresh frame, `/api/frame.jpg` returns HTTP **204 No Content**. This is a normal startup/reconnection state; a specific error is reported separately by `/api/status` and the camera page. If video itself is unavailable, test the complete URL in VLC, then close VLC and other camera viewers before reconnecting. If GPU processing fails, try **Processor → CPU**. Missing dependencies and failed weight downloads have their own error codes.

## What it measures

| Measurement | Output |
|---|---|
| Litter-box visits | Confirmed zone-entry count over the last 24 hours |
| Time in the litter-box zone | First-to-last observed presence per visit, plus time inside the rolling window |
| Food-station visits | Visits near the configured food station |
| Water-station visits | Visits near the configured water station |

Food and water entries are **station visits**, not confirmation of eating or drinking. Litter-box visits do not confirm urination or defecation. The application does not measure food or water volume, provide a diagnosis, or identify individual cats in a shared household.

This first version assumes **one cat uses the monitored stations**. Multiple visible cats or ambiguous food/water overlap pause attribution and can mark a visit uncertain. A different cat appearing alone cannot be identified as a different individual. Use separate monitored spaces for individual measurements.

## Original desktop interface on Windows

1. Install **Python 3.13** from [python.org](https://www.python.org/downloads/).
2. Extract this project and double-click **`start_windows.bat`**.
3. The first launch installs dependencies. A release with an included shared detector loads it locally. A source-only installation without release weights downloads a starter detector on its first load.
4. Paste the complete local RTSP URL into the connection dialog. It displays masked characters; the **Paste** button and **Show URL** checkbox make input easy to check.
5. Enter a cat name. Keep the current detector and processor `auto`.
6. Select the three station areas. Enter confirms each rectangle; C skips a station that is not visible.
7. Leave the zones empty for 8 seconds so the counters can arm.

Draw each station around the area the cat occupies while using it, rather than only the tiny bowl opening. Stations must not overlap. If food and water are adjacent and the same cat box covers both, separate them or adjust the camera angle and zones.

The zone selector always shows the complete camera frame, fitted to the desktop without cropping or stretching. Selections are mapped back to the camera's original coordinates. The monitor fits its video below a status panel whose text is drawn at display resolution, rather than shrinking the text with a large camera image.

To select the zones again, check **Select or change station zones** in the connection dialog. To update an existing installation, extract the updated project into the same project directory and replace the source files; the archive does not contain your `local/` records or `.venv/` installation.

Use a continuous RTSP feed and a fixed camera view. Disable automatic pan/tilt tracking; reselect zones after moving the camera or changing its view.

Press **Q** to stop. Press **S** to save the complete raw camera frame locally, without station rectangles or detection text. Press **D** to show other model classes and confidence labels.

The computer must remain awake and the application running. Camera outages, suspension, and long processing gaps create unobserved time. The dashboard shows observed coverage so a partial day is not presented as a fully observed 24 hours. Coverage indicates monitoring time, not proof that every visit was detected.

### Preview before collecting measurements

```powershell
.\start_windows.bat --preview
```

Preview shows detections without opening a visit database or recording coverage. Press S when a visible cat is missed. If Python/CUDA is already configured, `auto` checks the GPU and otherwise selects CPU. A GPU is optional. Prefer 640 or 960 if processing becomes slow; a larger model or larger image size can reduce throughput.

Video display and detection run independently. The display targets up to 30 FPS, while one model worker analyzes the newest available frame instead of queuing old footage. The panel shows measured **Video FPS**, **AI FPS**, and the selected processor. Actual video speed is limited by the camera stream, decoding, network, and computer; this does not create frames missing from the stream. Visit time and monitoring coverage advance only on new analyzed frames, never on repeated overlays.

**ByteTrack** associates detections with persistent track IDs. Your confidence setting gates new tracks; weaker matched detections may recover an existing track. A green `Cat #ID` box represents an observed track. When detection is briefly lost, the last box is held for up to **5 seconds**, displayed in orange with **held**. This is a visual estimate at the last known position; it is not a new observation and does not advance visit time.

To analyze the complete camera frame:

```powershell
.\start_windows.bat --full-frame
```

The setup dialog supports sizes up to 1920. Image size affects model processing; it does not increase the camera's native resolution.

## Records and the rolling 24-hour window

- SQLite databases are under `local/data/`, with a separate file for each cat name.
- CSV reports are under `local/reports/`. Times in CSV are UTC with explicit offsets.
- `python app.py --summary` prints the selected cat's latest 24-hour totals and exports its CSV.
- A visit counts in a window when its first observed entry is in that window.
- Observed duration is clipped to the window, including visits that began before it.
- An entry requires 2 seconds of evidence, with short gaps tolerated. Initial startup requires 8 seconds with clear zones.
- A lost detection is **unknown presence**, not a confirmed exit. An active visit stays the same when the cat is detected again, including after a track ID changes. Its last observed time does not advance while the cat is missing.
- A departure requires the observed cat to remain beyond a zone's 20% boundary margin for 3 seconds. This prevents ordinary bounding-box jitter from closing a visit.
- Gaps of 2 seconds or longer flag the retained visit uncertain. If the cat leaves entirely outside the visible camera area, a subsequent appearance may remain part of this uncertain visit because no departure was observed. This conservative rule prevents false extra visits but can merge real visits when the exit path is not visible. Use a view that includes approach and exit paths; disable the station focus crop when it hides them.
- A camera interruption closes an active record as interrupted and requires an observed departure before counting another visit at that station during the same run.
- Visits interrupted by a connection failure, shutdown, or attribution ambiguity retain their observations and an incomplete status.
- The SQLite database remains authoritative if Excel locks the CSV. Close Excel to refresh the export.

This is an observational prototype. Compare at least 10 manually observed visits, passes near stations, and empty scenes with its records before relying on the counts. Review incomplete visits.

## Privacy and camera access

- The RTSP URL and camera credentials remain in process memory for the session. They are not written to configuration, the database, CSV, training metadata, or source files.
- Only private local IP addresses, local hostnames resolving exclusively to those addresses, and authenticated URLs are accepted. Public internet endpoints are rejected.
- Requiring credentials in a URL does **not** prove that the camera rejects anonymous access; check authentication on the camera itself.
- The application pulls camera video as a client. Browser mode also runs an authenticated HTTP engine bound to loopback only. It does not change router settings or expose an inbound camera service to the network.
- Video is not recorded or uploaded. Desktop images are saved when S is pressed. Browser images are saved when imported or when the user enables automatic collection. Capture and training files stay under `local/`.
- Ultralytics analytics and crash reporting are disabled via `sync=False`. OpenCV/FFmpeg camera logging is suppressed to reduce URL exposure.
- Package installation requires internet. Source-only installations without included weights also download a starter detector. Inference and the app's measurement storage run locally.
- Authentication such as Digest protects login; ordinary RTSP does not necessarily encrypt the video. Keep the camera on your trusted local network and maintain its firmware and authentication settings.

These controls reduce exposure; they do not audit the camera firmware or the router's existing configuration.

## Improve cat detection

The developer trains one shared detector from the currently selected checkpoint. For scenes where it fails, collect varied, **manually reviewed** examples from actual cameras: different coats, distances, poses, daylight, infrared, partial occlusion, empty stations, dogs, and objects that create false positives. A fresh source-only installation can use a COCO-pretrained starter, which already includes cats.

The project includes developer frame review, manual labeling, session-based dataset preparation, and local fine-tuning. See [docs/SHARED_MODEL.md](docs/SHARED_MODEL.md) for the release workflow and [docs/TRAINING.md](docs/TRAINING.md) for labeling and infrared-camera guidance. This source update has no newly trained detector weights; end users receive the developer's evaluated model in a completed release.

For a public model, collect diverse cats and cameras with permission. A Mila-only fine-tune may help Mila while reducing performance on other cats. Keep the working baseline privately and evaluate diverse unseen recordings before selecting the shared release detector.

## Other desktop platforms

On a Python installation with Tk support, run `sh start.sh`, or create a virtual environment, install `requirements.txt`, and run `python app.py`. GUI camera operation has not been verified on every platform; this release targets Windows desktops.

## Tests

Core tests run without OpenCV, PyTorch, or camera access:

```powershell
python -m unittest discover -s tests -v
```

The optional simulated-monitor integration test uses NumPy and mocks the camera and detector. The real camera, detection quality, CUDA, installer, and native GUI must still be checked on the target machine.

## Shared detector for distribution

Use the current trained detector as the next training base. The web laboratory shows one selected detector, keeps evaluation session assignments stable, compares old and new cat-only checkpoints, and can include the selected detector in `assets/detector/`. New user installations automatically load that included model. See [the shared model workflow](docs/SHARED_MODEL.md).

A developer release needs real evaluated weights. This source ZIP contains the workflow, not the private Mila dataset or a newly trained shared detector. Test other cats and cameras before making general accuracy claims. End users do not need to label or train.

## Publish to GitHub

Keep the project source and documentation public. `local/`, `.venv/`, private weights, databases, videos, and CSV files are ignored. Explicitly bundled release weights under `assets/detector/` and `assets/behavior/` are allowed and checked against their manifests. The repository contains no real camera URL or credentials.

```powershell
git init
git add .
python tools/check_public_files.py
git commit -m "Add local cat station monitoring"
```

Create your GitHub repository, add its URL as the remote, and push. Re-run the public-file check before publishing later changes. The check is a guard against common mistakes, not a full security audit.

## License and references

This project is released under **GNU AGPL-3.0-only**, with the full license in `LICENSE`, consistent with the open-source use of Ultralytics. The pretrained models and library retain their upstream licensing. See [Ultralytics licensing](https://www.ultralytics.com/license).

- [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics training](https://docs.ultralytics.com/modes/train/)
- [Ultralytics validation](https://docs.ultralytics.com/modes/val/)
- [Ultralytics privacy controls](https://docs.ultralytics.com/help/privacy)
