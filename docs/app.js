'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const token = document.querySelector('meta[name="catcare-engine"]').content;
  const local = ['127.0.0.1', 'localhost'].includes(location.hostname) &&
    token.length > 25 && token !== '__ENGINE_TOKEN__';
  const state = { status: null, configLoaded: false, zones: {}, frame: null,
    images: [], sessions: [], galleryUrls: [], label: null, labelImage: null, boxes: [],
    drawing: null, cameraBusy: false, frameBusy: false, libraryBusy: false };
  const names = { overview: 'Overview', camera: 'Camera & zones', library: 'Cat library', training: 'Personalize detection' };
  const stationNames = { litter_box: 'Litter box', food: 'Food', water: 'Water' };
  const colors = { litter_box: '#9b7fc1', food: '#d0a45a', water: '#65a3bb' };
  let toastTimeout;

  function toast(message, error = false) {
    $('toast').textContent = message;
    $('toast').classList.toggle('error', error);
    $('toast').hidden = false;
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => { $('toast').hidden = true; }, error ? 9000 : 4500);
  }

  async function api(path, data, raw = false) {
    if (!local) throw new Error('Start the local engine and open the local app first.');
    const options = { headers: { 'X-CatCare-Token': token }, cache: 'no-store', credentials: 'same-origin' };
    if (data !== undefined) {
      options.method = 'POST';
      options.headers['Content-Type'] = raw ? 'application/octet-stream' : 'application/json';
      options.body = raw ? data : JSON.stringify(data);
    }
    const response = await fetch(path, options);
    if (!response.ok) {
      const result = await response.json().catch(() => ({}));
      throw new Error(result.error || 'The local engine could not complete that action.');
    }
    return response;
  }

  async function action(fn) {
    try { await fn(); } catch (error) { toast(error.message, true); }
  }

  function navigate() {
    const page = location.hash.slice(1) in names ? location.hash.slice(1) : 'overview';
    document.querySelectorAll('[data-view]').forEach(view => { view.hidden = view.dataset.view !== page; });
    document.querySelectorAll('[data-page]').forEach(link => {
      link.classList.toggle('active', link.dataset.page === page);
      if (link.dataset.page === page) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    $('page-name').textContent = names[page];
    if (page === 'library' && local) action(refreshLibrary);
    if (page === 'camera') drawCamera();
  }
  addEventListener('hashchange', navigate);
  $('setup-toggle').onclick = () => { $('setup-guide').hidden = !$('setup-guide').hidden; };
  document.querySelectorAll('[data-local]').forEach(element => { element.disabled = !local; });
  if (local) {
    $('engine-banner').hidden = true;
    $('engine-chip').classList.add('connected');
    $('engine-label').textContent = 'Local engine connected';
  }

  function loadConfig(config) {
    $('cat-name').value = config.cat_label;
    $('library-cat-name').value = config.cat_label;
    $('sidebar-name').textContent = config.cat_label === 'Cat' ? "Your cat's space" : `${config.cat_label}'s space`;
    $('device').value = config.device;
    $('image-size').value = config.image_size;
    $('confidence').value = config.confidence;
    const modelSelect = $('detector');
    if (!['yolo26n.pt', 'yolo26s.pt'].includes(config.model)) {
      const custom = document.createElement('option');
      custom.value = config.model;
      custom.textContent = 'Your trained detector';
      modelSelect.querySelectorAll('[data-custom]').forEach(node => node.remove());
      custom.dataset.custom = 'true';
      modelSelect.append(custom);
    }
    modelSelect.value = config.model;
    state.zones = structuredClone(config.zones || {});
    state.configLoaded = true;
  }

  function renderStatus(row) {
    state.status = row;
    if (!state.configLoaded) loadConfig(row.config);
    const active = row.running;
    $('camera-phase').textContent = { live: 'Live', connecting: 'Connecting', loading_model: 'Loading detector',
      preview: 'Preview only', error: 'Needs attention', offline: 'Not connected' }[row.phase] || row.phase;
    $('camera-message').textContent = row.message;
    $('camera-diagnostic').hidden = !row.error;
    if (row.error) {
      $('diagnostic-message').textContent = row.notice || row.error.message;
      $('diagnostic-detail').textContent = `${row.error.code} · ${row.error.stage}\n${row.error.detail}`;
    }
    $('camera-fps').textContent = `Camera ${(row.video_fps || 0).toFixed(1)} FPS · AI ${(row.ai_fps || 0).toFixed(1)} FPS${row.processor ? ` · ${row.processor}` : ''}`;
    const busy = ['preparing', 'training', 'evaluating'].includes(row.training.phase);
    $('connect-button').disabled = !local || active || busy || state.cameraBusy;
    $('disconnect-button').disabled = !local || !active || state.cameraBusy;
    $('save-zones').disabled = !local || active || !state.frame;
    $('collect-button').disabled = !local || !active;
    $('collect-button').textContent = row.collecting ? `Stop collecting · ${row.collected} saved` : 'Collect camera images automatically';
    $('collection-note').textContent = row.collecting ? 'Collecting varied camera frames. Collection also works when the cat is missed or misclassified.' :
      `Collected this camera session: ${row.collected || 0}. Up to 120 images. Review them before training.`;
    const summary = row.summary;
    for (const name of Object.keys(stationNames)) {
      const available = summary && row.config.zones[name];
      $(`metric-${name}`).textContent = available ? (summary.counts[name] || 0) : '—';
      $(`duration-${name}`).textContent = available ? `${((summary.durations[name] || 0) / 60).toFixed(1)} min · ${summary.uncertain[name] || 0} incomplete` : 'Zone not configured';
    }
    $('coverage').textContent = summary && Object.keys(row.config.zones).length ?
      `${(summary.observed_seconds / 3600).toFixed(2)} of 24 hours observed · missing detections may affect measurements` :
      'Connect a camera and select zones to begin recording.';
    renderTraining(row.training);
    drawCamera();
  }

  async function poll() {
    if (!local) return;
    try {
      renderStatus(await (await api('/api/status')).json());
      $('engine-chip').classList.add('connected');
      $('engine-label').textContent = 'Local engine connected';
    } catch {
      $('engine-chip').classList.remove('connected');
      $('engine-label').textContent = 'Local engine unavailable';
      $('connect-button').disabled = true;
      $('disconnect-button').disabled = true;
    } finally { setTimeout(poll, 1000); }
  }

  $('show-rtsp').onchange = () => { $('rtsp').type = $('show-rtsp').checked ? 'text' : 'password'; };
  $('copy-diagnostic').onclick = () => action(async () => {
    const row = state.status;
    if (!row?.error) return;
    const report = JSON.stringify({ version: row.version, phase: row.phase,
      processor: row.processor, error: row.error }, null, 2);
    if (!navigator.clipboard) throw new Error('Select and copy the diagnostic details shown above.');
    await navigator.clipboard.writeText(report);
    toast('Diagnostic copied. Camera credentials are excluded.');
  });
  $('camera-form').onsubmit = event => {
    event.preventDefault();
    action(async () => {
      if (state.cameraBusy) return;
      state.cameraBusy = true;
      $('connect-button').disabled = true;
      try {
        const config = { cat_label: $('cat-name').value, device: $('device').value,
          image_size: Number($('image-size').value), confidence: Number($('confidence').value) };
        if (['yolo26n.pt', 'yolo26s.pt'].includes($('detector').value)) config.model = $('detector').value;
        await api('/api/settings', config);
        await api('/api/connect', { url: $('rtsp').value.trim() });
        $('rtsp').value = ''; // Credentials live only in the running camera process.
        state.configLoaded = false;
        toast('Connecting locally. The first run may download detector weights.');
      } finally { state.cameraBusy = false; }
    });
  };
  $('disconnect-button').onclick = () => action(async () => {
    state.cameraBusy = true;
    $('disconnect-button').disabled = true;
    try { await api('/api/disconnect', {}); toast('Camera disconnected. You can draw zones on the last frame.'); }
    finally { state.cameraBusy = false; }
  });

  async function getFrame() {
    if (local && state.status?.running && !state.frameBusy && !document.hidden) {
      state.frameBusy = true;
      try {
        const response = await api('/api/frame.jpg');
        if (response.status !== 204) {
          const bitmap = await createImageBitmap(await response.blob());
          state.frame?.close();
          state.frame = bitmap;
          $('video-empty').hidden = true;
          drawCamera();
        }
      } catch { /* Status explains camera connection failures. */ }
      finally { state.frameBusy = false; }
    }
    if (local) setTimeout(getFrame, 50);
  }

  function rectangle(canvas, box, label, color, held = false) {
    const ctx = canvas.getContext('2d');
    const ratio = canvas.width / Math.max(1, canvas.getBoundingClientRect().width);
    const [a, b, c, d] = box;
    const x = a * canvas.width, y = b * canvas.height, w = (c - a) * canvas.width, h = (d - b) * canvas.height;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * ratio;
    if (held) ctx.setLineDash([5 * ratio, 4 * ratio]);
    ctx.strokeRect(x, y, w, h);
    if (label) {
      ctx.setLineDash([]);
      ctx.font = `600 ${12 * ratio}px system-ui`;
      const width = ctx.measureText(label).width + 12 * ratio;
      const top = Math.max(0, y - 23 * ratio);
      const left = Math.min(x, canvas.width - width);
      ctx.fillStyle = color;
      ctx.fillRect(left, top, width, 22 * ratio);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, left + 6 * ratio, top + 15 * ratio);
    }
    ctx.restore();
  }

  function drawCamera() {
    if (!state.frame || document.querySelector('[data-view="camera"]').hidden) return;
    const canvas = $('camera-canvas');
    canvas.width = state.frame.width;
    canvas.height = state.frame.height;
    canvas.getContext('2d').drawImage(state.frame, 0, 0);
    const row = state.status;
    const now = Date.now() / 1000;
    if (row?.running) {
      const tracks = (row.tracks || []).filter(track => now - track.last_seen <= 5);
      if (now - (row.prediction_at || 0) < 2) {
        for (const detection of row.detections || []) {
          if (detection.score < row.config.confidence || (detection.label === 'cat' && tracks.length)) continue;
          rectangle(canvas, detection.box, `${detection.label} ${detection.score.toFixed(2)}`,
            detection.label === 'cat' ? '#386d4d' : '#a16a35');
        }
      }
      for (const track of tracks) {
        const held = track.held || now - (row.prediction_at || 0) > 2;
        rectangle(canvas, track.box, held ? `Cat #${track.id} · last position` : `Cat #${track.id} ${track.score.toFixed(2)}`,
          held ? '#a16a35' : '#386d4d', held);
      }
    }
    for (const [name, box] of Object.entries(state.zones)) rectangle(canvas, box, stationNames[name], colors[name]);
    if (state.drawing?.mode === 'zone') rectangle(canvas, dragBox(state.drawing), '', colors[$('zone-select').value]);
  }

  function point(event, canvas) {
    const rect = canvas.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
  }
  function dragBox(drag) {
    return [Math.min(drag.start[0], drag.end[0]), Math.min(drag.start[1], drag.end[1]),
      Math.max(drag.start[0], drag.end[0]), Math.max(drag.start[1], drag.end[1])];
  }
  function installDrawing(canvas, mode, enabled, commit, redraw) {
    canvas.addEventListener('pointerdown', event => {
      if (!enabled()) return;
      canvas.setPointerCapture(event.pointerId);
      state.drawing = { mode, start: point(event, canvas), end: point(event, canvas) };
      event.preventDefault();
    });
    canvas.addEventListener('pointermove', event => {
      if (state.drawing?.mode !== mode) return;
      state.drawing.end = point(event, canvas); redraw();
    });
    canvas.addEventListener('pointerup', event => {
      if (state.drawing?.mode !== mode) return;
      state.drawing.end = point(event, canvas);
      const box = dragBox(state.drawing);
      if (box[2] - box[0] > .01 && box[3] - box[1] > .01) commit(box);
      state.drawing = null; redraw();
    });
    canvas.addEventListener('pointercancel', () => { state.drawing = null; redraw(); });
  }
  installDrawing($('camera-canvas'), 'zone', () => {
    if (!local || !$('zone-select').value || !state.frame) return false;
    if (state.status?.running) { toast('Disconnect first to edit zones on the last camera frame.'); return false; }
    return true;
  }, box => { state.zones[$('zone-select').value] = box; }, drawCamera);
  $('clear-zone').onclick = () => {
    if (state.status?.running) return toast('Disconnect before editing zones.');
    delete state.zones[$('zone-select').value]; drawCamera();
  };
  $('save-zones').onclick = () => action(async () => {
    await api('/api/zones', { zones: state.zones });
    toast('Station zones saved. Reconnect your camera to begin counting visits.');
  });
  addEventListener('resize', drawCamera);

  function revokeGallery() { state.galleryUrls.forEach(URL.revokeObjectURL); state.galleryUrls = []; }
  async function refreshLibrary() {
    if (state.libraryBusy) return;
    state.libraryBusy = true;
    try {
      const data = await (await api('/api/library')).json();
      state.images = data.images;
      state.sessions = data.sessions;
      const selected = $('session-select').value;
      $('session-select').replaceChildren(new Option('Choose a recording session', ''));
      for (const session of data.sessions) $('session-select').add(new Option(session.name, session.id));
      $('session-select').value = selected || data.sessions.at(-1)?.id || '';
      const reviewed = data.images.filter(image => image.reviewed).length;
      $('library-count').textContent = data.images.length;
      $('library-stats').textContent = `${data.images.length} images · ${reviewed} reviewed · ${data.sessions.length} sessions`;
      if (!data.images.length) {
        revokeGallery();
        $('photo-grid').replaceChildren();
        const empty = document.createElement('div');
        empty.className = 'empty-library';
        const title = document.createElement('strong'); title.textContent = 'Every cat starts somewhere.';
        const message = document.createElement('p'); message.textContent = 'Add photos or collect camera frames to start your library.';
        empty.append(title, message); $('photo-grid').append(empty);
        return;
      }
      revokeGallery();
      $('photo-grid').replaceChildren();
      for (const image of data.images) {
        const button = document.createElement('button');
        button.className = `photo-card${image.reviewed ? ' reviewed' : ''}`;
        button.type = 'button';
        button.setAttribute('aria-label', `Review image from ${image.session_name}`);
        const picture = document.createElement('img');
        picture.alt = `Learning image from ${image.session_name}`;
        picture.loading = 'lazy';
        const label = document.createElement('span');
        label.textContent = image.reviewed ? (image.boxes.length ? `✓ ${image.boxes.length} cat box(es)` : '✓ Empty scene') : 'Review image ↗';
        button.append(picture, label);
        button.onclick = () => action(() => openLabel(image));
        $('photo-grid').append(button);
        try {
          const url = URL.createObjectURL(await (await api(`/api/thumbnail/${image.id}`)).blob());
          state.galleryUrls.push(url); picture.src = url;
        } catch { label.textContent = 'Could not load image'; }
      }
    } finally { state.libraryBusy = false; }
  }
  $('new-session').onclick = () => action(async () => {
    const session = await (await api('/api/sessions', { name: $('session-name').value })).json();
    $('session-name').value = '';
    await refreshLibrary();
    $('session-select').value = session.id;
    toast('Recording session created. Keep images from the same recording together.');
  });
  $('save-profile').onclick = () => action(async () => {
    const changed = $('library-cat-name').value.trim() !== state.status?.config.cat_label;
    await api('/api/settings', { cat_label: $('library-cat-name').value });
    if (changed) { state.frame?.close(); state.frame = null; $('video-empty').hidden = false; }
    loadConfig((await (await api('/api/status')).json()).config);
    await refreshLibrary();
    toast('Cat profile selected. Photos and labels are kept separate for each name.');
  });
  async function importFiles(files) {
    if (!local) throw new Error('Open the local app before adding private images.');
    const session = $('session-select').value;
    if (!session) throw new Error('Choose or create a recording session first.');
    if (!files.length) return;
    $('photo-input').disabled = true;
    let count = 0;
    try {
      for (const file of files) {
        if (file.size > 12_000_000 || !['image/jpeg', 'image/png'].includes(file.type)) {
          toast('Skipped a file: use JPEG or PNG images up to 12 MB.', true); continue;
        }
        try { await api(`/api/upload/${session}`, file, true); count++; }
        catch (error) { toast(error.message, true); }
      }
      await refreshLibrary();
      toast(`${count} image${count === 1 ? '' : 's'} added locally.`);
    } finally { $('photo-input').disabled = false; $('photo-input').value = ''; }
  }
  $('photo-input').onchange = event => action(() => importFiles([...event.target.files]));
  for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, event => {
    event.preventDefault(); if (local) $('dropzone').classList.add('dragging');
  });
  $('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragging'));
  $('dropzone').addEventListener('drop', event => {
    event.preventDefault(); $('dropzone').classList.remove('dragging');
    action(() => importFiles([...event.dataTransfer.files]));
  });
  $('refresh-library').onclick = () => action(refreshLibrary);
  $('collect-button').onclick = () => action(async () => {
    await api('/api/collect', { enabled: !state.status?.collecting });
    toast(state.status?.collecting ? 'Automatic collection stopped. Refresh to review your images.' : 'Collecting learning frames locally.');
  });

  async function openLabel(image) {
    const bitmap = await createImageBitmap(await (await api(`/api/image/${image.id}`)).blob());
    state.labelImage?.close();
    state.label = image;
    state.labelImage = bitmap;
    state.boxes = structuredClone(image.boxes);
    if (!$('label-dialog').open) $('label-dialog').showModal();
    drawLabel();
  }
  function drawLabel() {
    if (!state.labelImage) return;
    const canvas = $('label-canvas');
    canvas.width = state.labelImage.width;
    canvas.height = state.labelImage.height;
    canvas.getContext('2d').drawImage(state.labelImage, 0, 0);
    state.boxes.forEach((box, index) => rectangle(canvas, box, `Cat ${index + 1}`, '#386d4d'));
    if (state.drawing?.mode === 'label') rectangle(canvas, dragBox(state.drawing), '', '#bc8e44');
    $('label-count').textContent = state.boxes.length ? `${state.boxes.length} cat box(es) drawn.` : 'No boxes drawn. Explicitly mark cat-free images as empty scenes.';
  }
  installDrawing($('label-canvas'), 'label', () => local && !!state.labelImage,
    box => { state.boxes.push(box); }, drawLabel);
  $('close-label').onclick = () => $('label-dialog').close();
  $('clear-boxes').onclick = () => { state.boxes = []; drawLabel(); };
  async function saveLabel(negative) {
    if (!state.label) return;
    if (!negative && !state.boxes.length) throw new Error('Draw at least one cat box, or choose No cat in this image.');
    await api('/api/labels', { id: state.label.id, boxes: negative ? [] : state.boxes, negative });
    const key = state.label.id;
    $('label-dialog').close();
    await refreshLibrary();
    const next = state.images.find(image => !image.reviewed && image.id !== key);
    toast('Reviewed label saved.');
    if (next) await openLabel(next);
  }
  $('save-label').onclick = () => action(() => saveLabel(false));
  $('empty-scene').onclick = () => action(() => saveLabel(true));

  function renderTraining(job) {
    const phase = job.phase || 'idle';
    const busy = ['preparing', 'training', 'evaluating'].includes(phase);
    const titles = { idle: 'Ready when your images are.', preparing: 'Preparing your images…',
      training: 'Training your model…', evaluating: 'Checking unseen sessions…', completed: 'Your detector is ready to review.',
      failed: 'This run needs a little attention.', cancelled: 'Training cancelled.' };
    $('training-title').textContent = titles[phase] || titles.idle;
    $('training-phase').textContent = { idle: 'Not started', preparing: 'Preparing dataset', training: 'Training locally',
      evaluating: 'Evaluating test sessions', completed: 'Trained and evaluated', failed: 'Run stopped', cancelled: 'Cancelled' }[phase] || phase;
    $('training-message').textContent = job.message || (phase === 'completed' ?
      'Review the held-out test results and compare real camera visits before replacing your current detector.' :
      busy ? 'Your computer is doing the work. Keep the local engine running.' :
      'Review all images from at least 3 independent recording sessions to begin.');
    $('training-epoch').textContent = job.epoch ? `Epoch ${job.epoch} / ${job.epochs}` : '—';
    $('training-progress').style.width = `${phase === 'completed' ? 100 : Math.min(95, 95 * (job.epoch || 0) / (job.epochs || 60))}%`;
    $('train-button').disabled = !local || busy || !!state.status?.running;
    $('cancel-training').disabled = !local || !busy;
    $('activate-model').hidden = phase !== 'completed';
    $('activate-model').disabled = !local || phase !== 'completed' || !!state.status?.running;
    $('training-results').hidden = phase !== 'completed';
    if (phase === 'completed') {
      $('training-results').replaceChildren();
      for (const [label, key] of [['Precision', 'precision'], ['Recall', 'recall'], ['mAP50', 'mAP50']]) {
        const panel = document.createElement('div');
        const small = document.createElement('small'); small.textContent = label;
        const value = document.createElement('strong'); value.textContent = `${((job[key] || 0) * 100).toFixed(1)}%`;
        panel.append(small, value); $('training-results').append(panel);
      }
    }
  }
  $('train-button').onclick = () => action(async () => {
    await api('/api/train', { epochs: Number($('epochs').value) });
    toast('Local training started. The previous detector remains selected.');
  });
  $('cancel-training').onclick = () => action(async () => {
    await api('/api/train/cancel', {}); toast('Training cancelled.');
  });
  $('activate-model').onclick = () => action(async () => {
    await api('/api/train/activate', {}); state.configLoaded = false;
    toast('Trained detector selected. Connect the camera to evaluate its real behavior.');
  });
  $('export-button').onclick = () => action(async () => {
    const url = URL.createObjectURL(await (await api('/api/report.csv')).blob());
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'cat-care-visits.csv';
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  addEventListener('beforeunload', () => { revokeGallery(); state.frame?.close(); state.labelImage?.close(); });
  navigate();
  poll();
  getFrame();
})();
