'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const token = document.querySelector('meta[name="catcare-engine"]').content;
  const local = ['127.0.0.1', 'localhost'].includes(location.hostname) &&
    token.length > 25 && token !== '__ENGINE_TOKEN__';
  const trainingMode = local && document.querySelector('meta[name="catcare-training"]')?.content === 'true';
  const state = { status: null, configLoaded: false, zones: {}, frame: null,
    images: [], sessions: [], galleryUrls: [], label: null, labelImage: null, boxes: [],
    drawing: null, cameraBusy: false, frameBusy: false, libraryBusy: false, bundleBusy: false };
  const names = { overview: 'Overview', camera: 'Camera & zones', library: 'Cat library', training: 'Train detection', behavior: 'Behavior lab' };
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
    let page = location.hash.slice(1) in names ? location.hash.slice(1) : 'overview';
    if (!trainingMode && ['library', 'training', 'behavior'].includes(page)) page = 'overview';
    document.querySelectorAll('[data-view]').forEach(view => { view.hidden = view.dataset.view !== page; });
    document.querySelectorAll('[data-page]').forEach(link => {
      link.classList.toggle('active', link.dataset.page === page);
      if (link.dataset.page === page) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    $('page-name').textContent = names[page];
    if (page === 'library' && local) action(refreshLibrary);
    if (page === 'behavior' && trainingMode) action(refreshBehaviorLibrary);
    if (page === 'camera') drawCamera();
  }
  addEventListener('hashchange', navigate);
  $('setup-toggle').onclick = () => { $('setup-guide').hidden = !$('setup-guide').hidden; };
  document.querySelectorAll('[data-local]').forEach(element => { element.disabled = !local; });
  document.querySelectorAll('[data-training]').forEach(element => { if (!trainingMode) element.hidden = true; });
  document.querySelectorAll('[data-consumer]').forEach(element => { element.hidden = trainingMode; });
  if (local) {
    $('engine-banner').hidden = true;
    $('engine-chip').classList.add('connected');
    $('engine-label').textContent = 'Local engine connected';
  }

  function loadConfig(config) {
    if (state.loadedProfile && state.loadedProfile !== config.cat_label) {
      behavior.readiness = null; behavior.captured = -1;
      $('behavior-data-summary').textContent = 'Profile changed. Open Behavior lab or press Check dataset to inspect this profile.';
      for (const id of ['behavior-data-issues', 'behavior-data-splits', 'behavior-data-periods']) $(id).replaceChildren();
    }
    state.loadedProfile = config.cat_label;
    $('cat-name').value = config.cat_label;
    $('library-cat-name').value = config.cat_label;
    $('sidebar-name').textContent = config.cat_label === 'Cat' ? "Your cat's space" : `${config.cat_label}'s space`;
    $('device').value = config.device;
    $('image-size').value = config.image_size;
    $('confidence').value = config.confidence;
    $('detector').value = config.model === 'bundled' ? 'WatchMyCat shared detector' :
      ['yolo26n.pt', 'yolo26s.pt'].includes(config.model) ? 'Starter detector · first training' : 'Current trained cat detector';
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
    const busy = [row.training, row.behavior_training].some(job => ['preparing', 'training', 'evaluating'].includes(job?.phase));
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
    renderBehavior(row);
    if (trainingMode) {
      $('fast-inference').disabled = busy || !local;
      const ms = row.processing_ms || 0;
      const lag = row.prediction_at ? Math.max(0, Date.now() / 1000 - row.prediction_at) : 0;
      $('inference-status').textContent = `${row.processor || 'Waiting for detector'} · AI input ${row.config.image_size} px` +
        (ms ? ` · ${ms.toFixed(0)} ms per processed frame · latest frame ${lag.toFixed(1)}s old` : '') +
        (row.processor === 'CPU' && row.running ? '. CPU is active; check GPU compatibility if you expected GPU processing.' : '');
    }
    drawCamera();
  }

  $('fast-inference').onclick = () => action(async () => {
    $('fast-inference').disabled = true;
    try {
      await api('/api/performance', { image_size: 640 });
      state.configLoaded = false;
      renderStatus(await (await api('/api/status')).json());
      toast('Your current detector is now using a 640 px AI input. Check that small or distant cats remain visible.');
    } finally { $('fast-inference').disabled = false; }
  });

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
        await api('/api/settings', config);
        await api('/api/connect', { url: $('rtsp').value.trim() });
        $('rtsp').value = ''; // Credentials live only in the running camera process.
        state.configLoaded = false;
        toast(state.status?.config.model === 'bundled' ? 'Connecting locally with the included shared detector.' : 'Connecting locally with the current detector.');
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
  function imageEditingLocked() {
    return !trainingMode || state.imagesMutating || state.labelSaving || [state.status?.training, state.status?.behavior_training]
      .some(job => ['preparing', 'training', 'evaluating'].includes(job?.phase));
  }
  const imageUrl = (image, thumbnail = false) => `/api/${thumbnail ? 'thumbnail' : 'image'}/${image.id}${image.trashed ? '?trash=1' : ''}`;
  function updateImageControls() {
    const locked = imageEditingLocked();
    $('image-empty-trash').disabled = locked || !state.imageTrash.length;
    $('image-empty-trash').textContent = `Empty image trash (${Math.min(600, state.imageTrash.length)})`;
    $('image-trash-current').hidden = !state.label || !!state.label.trashed;
    $('image-restore-current').hidden = !state.label?.trashed;
    $('image-trash-current').disabled = $('image-restore-current').disabled = locked;
    for (const id of ['clear-boxes', 'empty-scene', 'save-label']) {
      $(id).disabled = locked || !state.labelImage || !!state.label?.trashed;
    }
    document.querySelectorAll('[data-image-action]').forEach(button => { button.disabled = locked; });
  }
  async function refreshLibrary() {
    if (state.libraryBusy) { state.libraryPending = true; return; }
    state.libraryBusy = true;
    $('image-more').disabled = true;
    try {
      const data = await (await api('/api/library')).json();
      state.images = data.images.map(image => ({ ...image, trashed: false }));
      state.imageTrash = (data.trash || []).map(image => ({ ...image, trashed: true }));
      state.sessions = data.sessions;
      const selected = $('session-select').value;
      $('session-select').replaceChildren(new Option('Choose a recording session', ''));
      for (const session of data.sessions) $('session-select').add(new Option(session.name, session.id));
      $('session-select').value = data.sessions.some(session => session.id === selected) ? selected : data.sessions.at(-1)?.id || '';
      const reviewed = data.images.filter(image => image.reviewed).length;
      $('library-count').textContent = data.images.length;
      const filter = $('image-filter').value;
      const matches = (filter === 'trash' ? state.imageTrash : state.images).filter(image =>
        !['reviewed', 'unreviewed'].includes(filter) || image.reviewed === (filter === 'reviewed'));
      const images = matches.slice(-state.imageLimit).reverse();
      $('library-stats').textContent = `${data.images.length} active · ${reviewed} reviewed · ${state.imageTrash.length} in trash · showing ${images.length} / ${matches.length} matching images`;
      $('image-more').hidden = matches.length <= state.imageLimit;
      revokeGallery(); $('photo-grid').replaceChildren(); updateImageControls();
      if (!images.length) {
        const empty = document.createElement('div');
        empty.className = 'empty-library';
        const title = document.createElement('strong'); title.textContent = 'No images match this filter.';
        const message = document.createElement('p'); message.textContent = 'Choose Active images, add photos, or collect camera frames.';
        empty.append(title, message); $('photo-grid').append(empty);
      }
      for (const image of images) {
        const card = document.createElement('div'); card.className = 'image-card';
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
        const manage = document.createElement('button'); manage.type = 'button'; manage.className = 'button small';
        manage.dataset.imageAction = image.trashed ? 'restore' : 'trash';
        manage.textContent = image.trashed ? 'Restore image' : 'Move to trash'; manage.disabled = imageEditingLocked();
        manage.setAttribute('aria-label', `${manage.textContent}: image from ${image.session_name}`);
        manage.onclick = () => action(() => changeImages(manage.dataset.imageAction, [image.id]));
        card.append(button, manage); $('photo-grid').append(card);
        try {
          const url = URL.createObjectURL(await (await api(imageUrl(image, true))).blob());
          state.galleryUrls.push(url); picture.src = url;
        } catch { label.textContent = 'Could not load image'; }
      }
    } finally {
      state.libraryBusy = false; $('image-more').disabled = false; updateImageControls();
      if (state.libraryPending) { state.libraryPending = false; action(refreshLibrary); }
    }
  }
  $('image-filter').onchange = () => { state.imageLimit = 60; action(refreshLibrary); };
  $('image-more').onclick = () => { state.imageLimit += 60; action(refreshLibrary); };
  async function changeImages(operation, ids) {
    if (imageEditingLocked()) throw new Error('Wait for training or the current image action to finish.');
    state.imagesMutating = true; updateImageControls();
    try {
      await api(`/api/images/${operation}`, { ids });
      if (state.label && ids.includes(state.label.id)) $('label-dialog').close();
      await refreshLibrary();
      toast(operation === 'trash' ? 'Image and label moved to trash. Choose Show → Trash to restore them.' :
        operation === 'restore' ? 'Image restored with its original label and recording session.' : `${ids.length} image(s) and their labels permanently deleted.`);
    } finally { state.imagesMutating = false; updateImageControls(); }
  }
  $('image-empty-trash').onclick = () => action(async () => {
    const ids = state.imageTrash.slice(-600).map(image => image.id);
    if (!ids.length || !confirm(`Permanently delete these ${ids.length} trashed images and their labels? This cannot be undone. Earlier training snapshots and trained models retain their data.`)) return;
    await changeImages('purge', ids);
  });
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
    const generation = ++state.labelGeneration;
    state.labelImage?.close(); state.labelImage = null;
    state.label = image;
    state.boxes = structuredClone(image.boxes);
    if (!$('label-dialog').open) $('label-dialog').showModal();
    const canvas = $('label-canvas'); canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    $('label-count').textContent = 'Loading image…'; updateImageControls();
    try {
      const bitmap = await createImageBitmap(await (await api(imageUrl(image))).blob());
      if (generation !== state.labelGeneration || !$('label-dialog').open) { bitmap.close(); return; }
      state.labelImage = bitmap; drawLabel();
    } catch (error) {
      if (generation !== state.labelGeneration || !$('label-dialog').open) return;
      $('label-count').textContent = 'Preview unavailable. You can still move this image to trash.';
      throw error;
    } finally { updateImageControls(); }
  }
  function drawLabel() {
    if (!state.labelImage) return;
    const canvas = $('label-canvas');
    canvas.width = state.labelImage.width;
    canvas.height = state.labelImage.height;
    canvas.getContext('2d').drawImage(state.labelImage, 0, 0);
    state.boxes.forEach((box, index) => rectangle(canvas, box, `Cat ${index + 1}`, '#386d4d'));
    if (state.drawing?.mode === 'label') rectangle(canvas, dragBox(state.drawing), '', '#bc8e44');
    $('label-count').textContent = state.label?.trashed ? 'Trashed image: restore it before changing its boxes.' :
      state.boxes.length ? `${state.boxes.length} cat box(es) drawn. Use one per actual cat.` : 'No boxes drawn. Explicitly mark cat-free images as empty scenes.';
  }
  installDrawing($('label-canvas'), 'label', () => !imageEditingLocked() && !!state.labelImage && !state.label?.trashed,
    box => { state.boxes.push(box); }, drawLabel);
  $('close-label').onclick = () => $('label-dialog').close();
  $('label-dialog').addEventListener('close', () => {
    state.labelGeneration++; state.labelImage?.close(); state.labelImage = null; state.label = null;
    if (state.drawing?.mode === 'label') state.drawing = null;
  });
  $('image-trash-current').onclick = () => action(async () => {
    if (state.label && !state.label.trashed) await changeImages('trash', [state.label.id]);
  });
  $('image-restore-current').onclick = () => action(async () => {
    if (state.label?.trashed) await changeImages('restore', [state.label.id]);
  });
  $('clear-boxes').onclick = () => { state.boxes = []; drawLabel(); };
  async function saveLabel(negative) {
    if (imageEditingLocked() || !state.label || state.label.trashed) return;
    if (!negative && !state.boxes.length) throw new Error('Draw at least one cat box, or choose No cat in this image.');
    state.labelSaving = true; updateImageControls();
    try {
      const key = state.label.id;
      await api('/api/labels', { id: key, boxes: negative ? [] : state.boxes, negative });
      $('label-dialog').close(); await refreshLibrary();
      const next = state.images.find(image => !image.reviewed && image.id !== key);
      toast('Reviewed label saved.');
      if (next) await openLabel(next);
    } finally { state.labelSaving = false; updateImageControls(); }
  }
  $('save-label').onclick = () => action(() => saveLabel(false));
  $('empty-scene').onclick = () => action(() => saveLabel(true));

  function renderTraining(job) {
    updateImageControls();
    const phase = job.phase || 'idle';
    const busy = ['preparing', 'training', 'evaluating'].includes(phase);
    const titles = { idle: 'Ready when your images are.', preparing: 'Preparing your images…',
      training: 'Training your model…', evaluating: 'Checking unseen sessions…', completed: 'Your detector is ready to review.',
      failed: 'This run needs a little attention.', cancelled: 'Training cancelled.' };
    $('training-title').textContent = titles[phase] || titles.idle;
    $('training-phase').textContent = { idle: 'Not started', preparing: 'Preparing dataset', training: 'Training locally',
      evaluating: 'Evaluating test sessions', completed: 'Trained and evaluated', failed: 'Run stopped', cancelled: 'Cancelled' }[phase] || phase;
    $('training-message').textContent = job.message || (phase === 'completed' ?
      'Compare the previous and candidate detector on held-out images, then test real recordings before selecting or packaging it.' :
      busy ? 'Your computer is doing the work. Keep the local engine running.' :
      'Continue from the current detector using reviewed examples from varied cats and cameras. Training, validation, and test sessions stay separate.');
    $('detector-current').textContent = state.status?.detector_name || 'Current detector';
    $('bundle-detector').disabled = !local || state.bundleBusy || busy || !!state.status?.running ||
      ['preparing', 'training', 'evaluating'].includes(state.status?.behavior_training?.phase);
    $('training-epoch').textContent = job.epoch ? `Epoch ${job.epoch} / ${job.epochs}` : '—';
    $('training-progress').style.width = `${phase === 'completed' ? 100 : Math.min(95, 95 * (job.epoch || 0) / (job.epochs || 60))}%`;
    $('train-button').disabled = !local || busy || !!state.status?.running || ['preparing', 'training', 'evaluating'].includes(state.status?.behavior_training?.phase);
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
      if (job.baseline) {
        const table = document.createElement('table'); table.className = 'detector-comparison';
        const caption = document.createElement('caption'); caption.textContent = 'Same held-out image set · previous detector vs candidate';
        const head = table.createTHead().insertRow();
        for (const title of ['Metric', 'Previous', 'Candidate', 'Change']) {
          const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = title; head.append(cell);
        }
        const body = table.createTBody();
        for (const [label, key] of [['Precision', 'precision'], ['Recall', 'recall'], ['mAP50', 'mAP50'], ['mAP50–95', 'mAP50_95']]) {
          const line = body.insertRow();
          const delta = ((job[key] - job.baseline[key]) * 100);
          for (const value of [label, `${(job.baseline[key] * 100).toFixed(1)}%`, `${(job[key] * 100).toFixed(1)}%`, `${delta >= 0 ? '+' : ''}${delta.toFixed(1)} pp`])
            line.insertCell().textContent = value;
        }
        table.prepend(caption); $('training-results').append(table);
      }
    }
  }
  $('train-button').onclick = () => action(async () => {
    await api('/api/train', { epochs: Number($('epochs').value) });
    toast('Training a new candidate from the current detector. The active weights remain selected.');
  });
  $('cancel-training').onclick = () => action(async () => {
    await api('/api/train/cancel', {}); toast('Training cancelled.');
  });
  $('activate-model').onclick = () => action(async () => {
    await api('/api/train/activate', {}); state.configLoaded = false;
    toast('Trained detector selected. Connect the camera to evaluate its real behavior.');
  });
  $('bundle-detector').onclick = () => action(async () => {
    if (state.bundleBusy) return;
    state.bundleBusy = true;
    $('bundle-detector').disabled = true;
    try {
      const row = await (await api('/api/train/bundle', {})).json();
      $('detector-release-note').textContent = `Shared detector ${row.version} included in assets/detector. New installations load it automatically. Existing custom weights remain selected in this workspace.`;
      toast('Selected detector included in the release. Private training photos were not copied.');
    } finally { state.bundleBusy = false; $('bundle-detector').disabled = false; }
  });
  $('export-button').onclick = () => action(async () => {
    const url = URL.createObjectURL(await (await api('/api/report.csv')).blob());
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'cat-care-visits.csv';
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  const behavior = { clips: [], trash: [], sessions: [], urls: [], busy: false, clip: null, frames: [],
    index: 0, playing: false, timer: null, generation: 0, captured: -1, saving: false,
    mutating: false, collectBusy: false, refreshPending: false, limit: 60 };
  const activityNames = { eating: 'Eating', drinking: 'Drinking', other: 'Other', exclude: 'Cannot tell' };

  function behaviorEditingLocked() {
    return behavior.mutating || behavior.saving || [state.status?.training, state.status?.behavior_training]
      .some(job => ['preparing', 'training', 'evaluating'].includes(job?.phase));
  }
  function filteredBehaviorClips() {
    const filter = $('behavior-filter').value;
    const station = $('behavior-station-filter').value;
    return (filter === 'trash' ? behavior.trash : behavior.clips).filter(clip =>
      (station === 'both' || clip.station === station) &&
      (['all', 'trash'].includes(filter) || (filter === 'unreviewed' ? clip.label === null : clip.label === filter)));
  }
  function updateBehaviorCleanup() {
    const locked = !trainingMode || behaviorEditingLocked();
    const excluded = filteredBehaviorClips().filter(clip => !clip.trashed && clip.label === 'exclude');
    $('behavior-trash-excluded').disabled = locked || !excluded.length;
    $('behavior-trash-excluded').textContent = `Move Cannot tell to trash (${excluded.length})`;
    $('behavior-empty-trash').disabled = locked || !behavior.trash.length;
    $('behavior-empty-trash').textContent = `Empty trash (${Math.min(600, behavior.trash.length)})`;
    $('behavior-trash-clip').hidden = !behavior.clip || !!behavior.clip.trashed;
    $('behavior-restore-clip').hidden = !behavior.clip?.trashed;
    $('behavior-trash-clip').disabled = $('behavior-restore-clip').disabled = locked;
    document.querySelectorAll('[data-clip-action]').forEach(button => { button.disabled = locked; });
    document.querySelectorAll('[data-behavior-label]').forEach(button => {
      button.hidden = !!behavior.clip?.trashed;
      button.disabled = locked || !behavior.frames.length;
    });
  }
  const behaviorFrameUrl = (clip, index) => `/api/behavior/frame/${clip.id}/${index}${clip.trashed ? '?trash=1' : ''}`;

  function renderBehavior(row) {
    const status = row.behavior || {};
    const job = row.behavior_training || { phase: 'idle' };
    const busy = ['preparing', 'training', 'evaluating'].includes(job.phase);
    const detectorBusy = ['preparing', 'training', 'evaluating'].includes(row.training?.phase);
    const labels = { waiting: 'Waiting for a sequence', candidate: 'Checking activity', away: 'Away from station',
      other: 'Nearby · other activity', uncertain: 'Unclear · time paused',
      probable_eating: 'Probably eating', probable_drinking: 'Probably drinking' };
    $('activity-message').textContent = !row.activity_model_available ?
      'Eating and drinking recognition is not available in this version. Station visits remain available.' :
      status.phase === 'error' ? `Activity recognition paused: ${status.error || 'Could not load the activity model.'}` :
      status.phase === 'loading' ? 'Preparing activity recognition…' :
      row.running ? 'Experimental activity estimates · only observed activity time is accumulated.' :
      'Connect your camera to observe activity. Previous records remain saved.';
    for (const station of ['food', 'water']) {
      const metric = status.summary?.[station];
      const available = row.activity_model_available && metric && row.config.zones[station];
      $(`activity-${station}-count`).textContent = available ? metric.sessions : '—';
      $(`activity-${station}-time`).textContent = available ? `${(metric.observed_seconds / 60).toFixed(1)} observed min · ${metric.incomplete} incomplete` : 'No activity observations';
      $(`activity-${station}-state`).textContent = !row.activity_model_available ? 'Not available' :
        !row.config.zones[station] ? 'Zone not configured' : !row.running ? 'Camera disconnected' : labels[status.stations?.[station]] || 'Waiting';
    }
    $('activity-export').disabled = !local || !status.summary || state.reportsBusy;
    if (!trainingMode) return;
    $('behavior-collect').disabled = !row.running || busy || detectorBusy || behavior.collectBusy;
    $('behavior-collect').textContent = status.collecting ? 'Stop collecting clips' : 'Collect behavior clips';
    $('behavior-capture-station').disabled = !local || status.collecting || behavior.collectBusy || busy || detectorBusy;
    $('behavior-session').disabled = !local || status.collecting || behavior.collectBusy || busy || detectorBusy;
    $('behavior-new-session').disabled = !local || status.collecting || behavior.collectBusy || busy || detectorBusy;
    for (const option of $('behavior-capture-station').options) {
      option.disabled = option.value === 'both' ? !row.config.zones.food && !row.config.zones.water : !row.config.zones[option.value];
    }
    if (status.collecting) $('behavior-capture-station').value = status.collection_stations?.length === 1 ? status.collection_stations[0] : 'both';
    const areas = status.collecting ? `Collecting: ${(status.collection_stations || []).map(station => stationNames[station]).join(' + ')}. ` : '';
    $('behavior-collection-note').textContent = `${areas}${status.captured || 0} clips saved this camera run. ${status.note || 'Captures cropped station sequences, including empty scenes. Review every clip.'}`;
    $('behavior-phase').textContent = job.phase === 'idle' ? 'Not started' : job.phase;
    $('behavior-training-message').textContent = job.message || (job.phase === 'completed' ?
      'Review test clips and validate complete unseen recordings before including this candidate in a release.' :
      busy ? 'Training locally. Keep the engine running.' : 'Review clips from at least 3 independent periods, including all activities and Other examples at both stations.');
    $('behavior-epoch').textContent = job.epoch ? `Epoch ${job.epoch} / ${job.epochs}` : job.prepared ? `Preparing ${job.prepared} / ${job.total} clips` : '—';
    $('behavior-progress').style.width = `${job.phase === 'completed' ? 100 : job.phase === 'preparing' ? 10 * (job.prepared || 0) / (job.total || 1) : 10 + 85 * (job.epoch || 0) / (job.epochs || 60)}%`;
    $('behavior-train').disabled = !local || row.running || busy || detectorBusy || !behavior.readiness?.ready || behavior.checkBusy;
    $('behavior-check-data').disabled = !local || behavior.checkBusy;
    $('behavior-run-requirement').textContent = row.running ? 'Disconnect the camera to enable training after the dataset checklist is complete.' :
      busy || detectorBusy ? 'Wait for the current training job to finish.' :
      !behavior.readiness?.ready ? 'Training is blocked by the dataset checklist above. Press Check dataset after fixing labels or periods.' : 'Camera disconnected. The dataset checklist permits training.';
    $('behavior-cancel').disabled = !busy;
    $('behavior-activate').hidden = job.phase !== 'completed';
    $('behavior-activate').disabled = row.running || busy || detectorBusy;
    $('behavior-bundled').disabled = row.running || busy || detectorBusy;
    $('behavior-results').hidden = job.phase !== 'completed';
    if (job.phase === 'completed') {
      $('behavior-results').replaceChildren();
      const headline = document.createElement('p');
      headline.textContent = `Held-out clips: ${job.clips || 0} · Macro F1: ${((job.macro_f1 || 0) * 100).toFixed(1)}%`;
      $('behavior-results').append(headline);
      for (const label of ['eating', 'drinking', 'other']) {
        const item = document.createElement('p'); const score = job.per_class?.[label] || {};
        item.textContent = `${activityNames[label]} · precision ${((score.precision || 0) * 100).toFixed(1)}% · recall ${((score.recall || 0) * 100).toFixed(1)}% · ${score.examples || 0} test clips`;
        $('behavior-results').append(item);
      }
    }
    if (behavior.captured !== (status.captured || 0) && location.hash === '#behavior') action(refreshBehaviorLibrary);
    behavior.captured = status.captured || 0;
    updateBehaviorCleanup();
  }

  async function refreshBehaviorLibrary() {
    if (!trainingMode) return;
    if (behavior.busy) { behavior.refreshPending = true; return; }
    behavior.busy = true;
    $('behavior-more').disabled = true;
    try {
      const data = await (await api('/api/behavior/library')).json();
      behavior.clips = data.clips.map(clip => ({ ...clip, trashed: false })).sort((a, b) => a.timestamps[0] - b.timestamps[0]);
      behavior.trash = (data.trash || []).map(clip => ({ ...clip, trashed: true })).sort((a, b) => a.timestamps[0] - b.timestamps[0]);
      behavior.sessions = data.sessions;
      if (data.readiness) renderBehaviorReadiness(data.readiness);
      const selected = $('behavior-session').value;
      $('behavior-session').replaceChildren(new Option('Choose a real recording period', ''));
      for (const session of data.sessions) $('behavior-session').add(new Option(session.name, session.id));
      $('behavior-session').value = data.sessions.some(session => session.id === selected) ? selected : data.sessions.at(-1)?.id || '';
      const reviewed = data.clips.filter(clip => clip.label !== null).length;
      behavior.urls.forEach(URL.revokeObjectURL); behavior.urls = [];
      $('behavior-clips').replaceChildren();
      const matches = filteredBehaviorClips();
      const clips = matches.slice(-behavior.limit).reverse();
      $('behavior-stats').textContent = `${data.clips.length} active · ${reviewed} reviewed · ${behavior.trash.length} in trash · showing ${clips.length} / ${matches.length} matching clips`;
      $('behavior-more').hidden = matches.length <= behavior.limit;
      updateBehaviorCleanup();
      if (!clips.length) {
        const text = document.createElement('p'); text.textContent = 'No clips match this filter. Collect a recording period or choose All clips.';
        $('behavior-clips').append(text);
      }
      for (const clip of clips) {
        const card = document.createElement('div'); card.className = 'behavior-clip-card';
        const button = document.createElement('button'); button.type = 'button'; button.className = 'photo-card';
        const image = document.createElement('img'); image.alt = `Review ${stationNames[clip.station]} clip`; image.loading = 'lazy';
        const label = document.createElement('span'); label.textContent = `${activityNames[clip.label] || 'Review'} · ${stationNames[clip.station]} · ${clip.duration.toFixed(1)}s`;
        button.append(image, label); button.onclick = () => action(() => openBehaviorClip(clip));
        const manage = document.createElement('button'); manage.type = 'button';
        manage.className = 'button small'; manage.dataset.clipAction = clip.trashed ? 'restore' : 'trash';
        manage.textContent = clip.trashed ? 'Restore clip' : 'Move to trash'; manage.disabled = behaviorEditingLocked();
        manage.setAttribute('aria-label', `${manage.textContent}: ${stationNames[clip.station]} clip from ${clip.session_name}`);
        manage.onclick = () => action(() => changeBehaviorClips(manage.dataset.clipAction, [clip.id]));
        card.append(button, manage); $('behavior-clips').append(card);
        try {
          const url = URL.createObjectURL(await (await api(behaviorFrameUrl(clip, 0))).blob());
          behavior.urls.push(url); image.src = url;
        } catch { image.alt = 'Preview unavailable. Open the clip or move it to trash.'; }
      }
    } finally {
      behavior.busy = false; $('behavior-more').disabled = false; updateBehaviorCleanup();
      if (behavior.refreshPending) { behavior.refreshPending = false; action(refreshBehaviorLibrary); }
    }
  }
  for (const id of ['behavior-filter', 'behavior-station-filter']) {
    $(id).onchange = () => { behavior.limit = 60; action(refreshBehaviorLibrary); };
  }
  $('behavior-refresh').onclick = () => action(refreshBehaviorLibrary);
  $('behavior-more').onclick = () => { behavior.limit += 60; action(refreshBehaviorLibrary); };
  async function changeBehaviorClips(operation, ids) {
    if (behaviorEditingLocked()) throw new Error('Wait for training or the current action to finish.');
    behavior.mutating = true; updateBehaviorCleanup();
    try {
      await api(`/api/behavior/clips/${operation}`, { ids });
      if (behavior.clip && ids.includes(behavior.clip.id)) $('behavior-dialog').close();
      await refreshBehaviorLibrary();
      toast(operation === 'trash' ? `${ids.length} clip(s) moved to trash. Choose Show → Trash to restore them.` :
        operation === 'restore' ? 'Clip restored with its original recording period and label.' : `${ids.length} trashed clip(s) permanently deleted.`);
    } finally { behavior.mutating = false; updateBehaviorCleanup(); }
  }
  $('behavior-trash-excluded').onclick = () => action(async () => {
    const ids = filteredBehaviorClips().filter(clip => !clip.trashed && clip.label === 'exclude').map(clip => clip.id);
    if (ids.length) await changeBehaviorClips('trash', ids);
  });
  $('behavior-empty-trash').onclick = () => action(async () => {
    const ids = behavior.trash.slice(-600).map(clip => clip.id);
    if (!ids.length || !confirm(`Permanently delete these ${ids.length} trashed clips and their frames? This cannot be undone. Active clips and trained models are kept.`)) return;
    await changeBehaviorClips('purge', ids);
  });
  $('behavior-new-session').onclick = () => action(async () => {
    const session = await (await api('/api/behavior/sessions', { name: $('behavior-session-name').value })).json();
    $('behavior-session-name').value = ''; await refreshBehaviorLibrary(); $('behavior-session').value = session.id;
  });
  $('behavior-collect').onclick = () => action(async () => {
    if (behavior.collectBusy) return;
    const enabled = !state.status?.behavior?.collecting;
    if (enabled && !$('behavior-session').value) throw new Error('Create or choose the real recording period first.');
    const station = $('behavior-capture-station').value;
    behavior.collectBusy = true; if (state.status) renderBehavior(state.status);
    try {
      await api('/api/behavior/collect', { enabled, session: $('behavior-session').value, station });
      if (state.status?.behavior) {
        state.status.behavior.collecting = enabled;
        state.status.behavior.collection_stations = enabled ? station === 'both' ? ['food', 'water'].filter(name => state.status.config.zones[name]) : [station] : [];
      }
      toast(enabled ? `Collecting ${station === 'both' ? 'configured station' : stationNames[station]} clips locally. Labels remain unreviewed.` : 'Clip collection stopped.');
    } finally { behavior.collectBusy = false; if (state.status) renderBehavior(state.status); }
  });
  function stopBehaviorPlayback() {
    clearTimeout(behavior.timer); behavior.timer = null; behavior.playing = false;
  }
  function drawBehaviorFrame() {
    const image = behavior.frames[behavior.index]; if (!image) return;
    const canvas = $('behavior-canvas'); canvas.width = image.width; canvas.height = image.height;
    canvas.getContext('2d').drawImage(image, 0, 0);
    $('behavior-scrub').value = behavior.index;
    $('behavior-frame-number').textContent = `Frame ${behavior.index + 1} / ${behavior.frames.length}`;
    $('behavior-play').textContent = behavior.playing ? 'Pause' : 'Play';
  }
  function playBehavior() {
    drawBehaviorFrame();
    if (!behavior.playing || !$('behavior-dialog').open) return;
    const stamps = behavior.clip?.timestamps || [];
    const delay = behavior.index + 1 < stamps.length ? Math.max(50, Math.min(1000, 1000 * (stamps[behavior.index + 1] - stamps[behavior.index]))) : 250;
    behavior.timer = setTimeout(() => { behavior.index = (behavior.index + 1) % behavior.frames.length; playBehavior(); }, delay);
  }
  async function openBehaviorClip(clip) {
    stopBehaviorPlayback(); behavior.generation += 1; const generation = behavior.generation;
    behavior.frames.forEach(frame => frame.close()); behavior.frames = []; behavior.clip = clip; behavior.index = 0;
    updateBehaviorCleanup();
    $('behavior-clip-description').textContent = `${clip.session_name} · ${stationNames[clip.station]} · loading…`;
    if (!$('behavior-dialog').open) $('behavior-dialog').showModal();
    const frames = [];
    try {
      for (let start = 0; start < clip.frames; start += 4) {
        const chunk = await Promise.allSettled(Array.from({ length: Math.min(4, clip.frames - start) }, async (_, offset) =>
          createImageBitmap(await (await api(behaviorFrameUrl(clip, start + offset))).blob())));
        for (const item of chunk) if (item.status === 'fulfilled') frames.push(item.value);
        const failed = chunk.find(item => item.status === 'rejected');
        if (failed) throw failed.reason;
      }
    } catch (error) {
      frames.forEach(frame => frame.close());
      if (generation !== behavior.generation || !$('behavior-dialog').open) return;
      $('behavior-clip-description').textContent = `${clip.session_name} · Preview could not be loaded. You can still move a damaged clip to trash.`;
      throw error;
    }
    if (generation !== behavior.generation || !$('behavior-dialog').open) { frames.forEach(frame => frame.close()); return; }
    behavior.frames = frames; behavior.playing = true;
    updateBehaviorCleanup();
    $('behavior-clip-description').textContent = `${clip.session_name} · ${stationNames[clip.station]} · ${clip.duration.toFixed(1)} seconds · ${activityNames[clip.label] || 'Not reviewed'}`;
    playBehavior();
  }
  $('behavior-close').onclick = () => $('behavior-dialog').close();
  $('behavior-dialog').addEventListener('close', () => {
    stopBehaviorPlayback(); behavior.generation += 1; behavior.frames.forEach(frame => frame.close()); behavior.frames = []; behavior.clip = null;
  });
  $('behavior-play').onclick = () => { if (!behavior.frames.length) return; const playing = behavior.playing; stopBehaviorPlayback(); behavior.playing = !playing; playBehavior(); };
  $('behavior-scrub').oninput = () => { stopBehaviorPlayback(); behavior.index = Number($('behavior-scrub').value); drawBehaviorFrame(); };
  async function saveBehaviorLabel(label) {
    if (behaviorEditingLocked() || !behavior.clip || behavior.clip.trashed || !behavior.frames.length) return;
    behavior.saving = true;
    updateBehaviorCleanup();
    try {
      const key = behavior.clip.id;
      await api('/api/behavior/labels', { id: key, label });
      $('behavior-dialog').close(); await refreshBehaviorLibrary();
      const next = behavior.clips.find(clip => clip.label === null && clip.id !== key);
      if (next) await openBehaviorClip(next);
    } finally { behavior.saving = false; updateBehaviorCleanup(); }
  }
  $('behavior-trash-clip').onclick = () => action(async () => {
    if (behavior.clip && !behavior.clip.trashed) await changeBehaviorClips('trash', [behavior.clip.id]);
  });
  $('behavior-restore-clip').onclick = () => action(async () => {
    if (behavior.clip?.trashed) await changeBehaviorClips('restore', [behavior.clip.id]);
  });
  document.querySelectorAll('[data-behavior-label]').forEach(button => {
    button.onclick = () => action(() => saveBehaviorLabel(button.dataset.behaviorLabel));
  });
  document.addEventListener('keydown', event => {
    if (!$('behavior-dialog').open || event.repeat) return;
    const label = { e: 'eating', d: 'drinking', o: 'other', x: 'exclude' }[event.key.toLowerCase()];
    if (label) { event.preventDefault(); action(() => saveBehaviorLabel(label)); }
  });
  $('behavior-train').onclick = () => action(async () => {
    const report = await checkBehaviorData();
    if (!report?.ready) throw new Error(report?.issues?.[0] || 'Check the behavior dataset before training.');
    await api('/api/behavior/train', { epochs: Number($('behavior-epochs').value) }); toast('Activity training started locally.');
  });
  $('behavior-cancel').onclick = () => action(async () => { await api('/api/behavior/train/cancel', {}); });
  $('behavior-activate').onclick = () => action(async () => {
    await api('/api/behavior/train/activate', {}); state.configLoaded = false; toast('Candidate selected for local evaluation.');
  });
  $('behavior-bundled').onclick = () => action(async () => {
    await api('/api/behavior/model/bundled', {}); state.configLoaded = false; toast('Included activity model selected.');
  });
  $('activity-export').onclick = () => action(async () => {
    const url = URL.createObjectURL(await (await api('/api/behavior/report.csv')).blob());
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'cat-care-activities.csv';
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  addEventListener('beforeunload', () => { revokeGallery(); state.frame?.close(); state.labelImage?.close(); stopBehaviorPlayback(); behavior.urls.forEach(URL.revokeObjectURL); behavior.frames.forEach(frame => frame.close()); });
  navigate();
  poll();
  getFrame();
})();
