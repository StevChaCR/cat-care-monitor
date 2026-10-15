# Publish the Cat Care frontend

The static frontend is in `docs/`. It uses relative asset paths and supports both a user-site root and a project subdirectory. It needs no Node build, CDN scripts, or API key.

## Publish

1. Add this project's source to your GitHub repository. Preserve the `.gitignore`; never add `local/`, `.venv/`, real RTSP URLs, photos, or model weights.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select your source branch, normally `main`, and **`/docs`**. Save.
5. GitHub displays the deployed site URL after deployment finishes.

The repository root `README.md` describes the Python application; `docs/index.html` is the public site's entry point. Keep `.nojekyll` in `docs/`.

## What visitors can do

On the public page visitors can explore the interface, read the local setup steps, and open **http://127.0.0.1:8765** after starting the engine on their own computer.

They run `start_web_windows.bat` once to install the requirements and open the local app. Subsequent launches use the same environment. Python 3.13 must be installed; a packaged Windows executable is not part of this release.

The local app enables RTSP input, private photo import, manual image review, station configuration, detection, records, and model training. This is a local application with a public frontend entry point, not an entirely browser-only RTSP service.

GitHub Pages cannot connect directly to a household RTSP camera or execute the Python training worker. The local engine serves its own copy of the interface. It deliberately rejects requests from the public site's origin; camera passwords are entered only in the local app. No CORS exception, router port forwarding, or public camera address is needed.

## Learning workflow

1. In **Cat library**, save the cat profile before importing photos. Use the same name to return to its images and selected model later.
2. Create a session for one real recording period and import its JPEG/PNG frames. Keep neighboring frames in that session. Alternatively, connect the camera and enable automatic image collection. A new camera connection creates a new collection session, but reconnecting the same recording does not make its footage independent.
3. Open each image, draw boxes around all visible cats, or explicitly mark a cat-free scene. Labels use the original full-frame coordinates.
4. Gather different recording periods, including difficult poses and empty scenes. At least three independent labeled sessions are required, and the dataset helper checks exact duplicates and held-out split coverage. More independent sessions give a more informative evaluation.
5. Disconnect the camera, then open **Personalize detection** and start training. Training executes locally in a child Python process; progress comes from real training callbacks. Early stopping can finish before the maximum epoch count. It is not a fixed-duration animation.
6. Review test precision, recall, and mAP50. The previous model remains selected until **Use this model** is clicked. Compare false detections, actual visits, and daytime/infrared footage against your baseline. The UI does not automatically certify a quality improvement.

No model was fine-tuned or certified for the user's camera as part of this frontend release. Uploaded photos are not automatically used as trusted labels. Individual cat identification and remote monitoring are separate future features.

## Verification scope

The project has unit and local HTTP integration checks for authentication, rejected foreign origins, private file isolation, image import/review, profile separation, and dataset gates. The web camera runtime shares the existing station geometry, visit state machine, camera reader, and ByteTrack worker. A native Windows run, a live Eufy stream, browser layout verification, and an end-to-end GPU training session still require testing on the user's computer.

Version 0.2.1 adds camera-first startup, preview-only recovery when AI fails, a single CPU retry for CUDA failures, and redacted diagnostics. A cached YOLO model was exercised on CPU through the local HTTP API using simulated camera frames; preview was available before the first prediction. This does not certify connectivity to a real RTSP camera. No-frame polling returns 204, while specific failures remain visible in the UI.
