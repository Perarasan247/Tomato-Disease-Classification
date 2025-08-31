const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const predLabelEl = document.getElementById('predLabel');
const healthyTagEl = document.getElementById('healthyTag');
const topkEl = document.getElementById('topk');
const probsEl = document.getElementById('probs');

let file = null;

function setStatus(msg) {
  statusEl.textContent = msg || '';
}

function bytesToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function showPreview(f) {
  bytesToDataURL(f).then(url => {
    preview.src = url;
    preview.style.display = 'block';
  });
}

function pickFile() { fileInput.click(); }

dropzone.addEventListener('click', pickFile);
fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    file = e.target.files[0];
    showPreview(file);
    predictBtn.disabled = false;
    setStatus('');
  }
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault(); e.stopPropagation();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  if (dt.files && dt.files[0]) {
    file = dt.files[0];
    showPreview(file);
    predictBtn.disabled = false;
    setStatus('');
  }
});

async function predict() {
  if (!file) return;

  setStatus('Uploading and running inference...');
  resultEl.classList.add('hidden');
  topkEl.innerHTML = '';
  probsEl.innerHTML = '';

  const form = new FormData();
  form.append('file', file, file.name);

  try {
    // ✅ Backend is on localhost:8000
    const res = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: form
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();

    // Headline
    predLabelEl.textContent = data.pred_class.replace('Tomato_', '').replaceAll('__','_');
    healthyTagEl.className = 'tag ' + (data.is_healthy ? 'good' : 'bad');
    healthyTagEl.textContent = data.is_healthy ? 'HEALTHY' : 'DISEASE';

    // Top-k
    data.topk.forEach((item) => {
      const row = document.createElement('div'); row.className = 'row';
      const name = document.createElement('div'); name.className = 'code';
      name.textContent = item.class_name.replace('Tomato_', '').replaceAll('__','_');
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('div'); fill.style.width = `${(item.prob*100).toFixed(1)}%`;
      bar.appendChild(fill);
      const pct = document.createElement('div'); pct.className = 'code';
      pct.textContent = `${(item.prob*100).toFixed(1)}%`;
      row.appendChild(name); row.appendChild(bar); row.appendChild(pct);
      topkEl.appendChild(row);
    });

    // All probs
    const entries = Object.entries(data.probs).sort((a,b)=>b[1]-a[1]);
    entries.forEach(([name, p]) => {
      const row = document.createElement('div'); row.className = 'row';
      const n = document.createElement('div'); n.className = 'code';
      n.textContent = name.replace('Tomato_', '').replaceAll('__','_');
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('div'); fill.style.width = `${(p*100).toFixed(1)}%`;
      bar.appendChild(fill);
      const pct = document.createElement('div'); pct.className = 'code';
      pct.textContent = `${(p*100).toFixed(1)}%`;
      row.appendChild(n); row.appendChild(bar); row.appendChild(pct);
      probsEl.appendChild(row);
    });

    resultEl.classList.remove('hidden');
    setStatus('Done.');
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  }
}

predictBtn.addEventListener('click', predict);
