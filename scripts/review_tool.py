"""
OCR Pseudo-Label Review Tool
=============================
Chạy: python review_tool.py --input data/processed/ocr/reviewed.jsonl
Mở trình duyệt: http://localhost:5000
"""

import argparse
import json
import base64
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import webbrowser

# ── Load data ────────────────────────────────────────────────────────────────

def load_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_records(records, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def image_to_base64(img_path: str) -> str:
    """Đọc ảnh và encode base64 để nhúng vào HTML."""
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        ext = Path(img_path).suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""

# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(records):
    total = len(records)
    done  = sum(1 for r in records if r.get("ground_truth_text", "").strip())
    buckets = {}
    fields  = {}
    for r in records:
        b = r.get("review_bucket", "unknown")
        buckets[b] = buckets.get(b, 0) + 1
        c = r.get("class", "unknown")
        fields[c] = fields.get(c, 0) + 1
    return {"total": total, "done": done, "remaining": total - done,
            "buckets": buckets, "fields": fields}

# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Review Tool — CCCD</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Thai:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0d0f14;
    --surface:  #161920;
    --card:     #1e2230;
    --border:   #2a2f3d;
    --accent:   #4ade80;
    --accent2:  #38bdf8;
    --warn:     #fb923c;
    --danger:   #f87171;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans Thai', sans-serif;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--sans);
         min-height: 100vh; }

  /* ── Layout ── */
  .layout { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }

  /* ── Sidebar ── */
  .sidebar { background: var(--surface); border-right: 1px solid var(--border);
             padding: 24px 16px; position: sticky; top: 0; height: 100vh;
             overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }
  .sidebar-title { font-family: var(--mono); font-size: 11px; font-weight: 600;
                   color: var(--accent); letter-spacing: 2px; text-transform: uppercase; }
  .stat-block { background: var(--card); border: 1px solid var(--border);
                border-radius: 8px; padding: 12px; }
  .stat-label { font-size: 11px; color: var(--muted); font-family: var(--mono);
                text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .stat-value { font-size: 26px; font-weight: 600; font-family: var(--mono); }
  .stat-value.green  { color: var(--accent); }
  .stat-value.blue   { color: var(--accent2); }
  .stat-value.orange { color: var(--warn); }

  .progress-wrap { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; }
  .progress-bar  { height: 100%; background: var(--accent);
                   transition: width .4s ease; border-radius: 4px; }

  .bucket-row { display: flex; justify-content: space-between; align-items: center;
                padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
  .bucket-row:last-child { border-bottom: none; }
  .badge { font-family: var(--mono); font-size: 11px; padding: 2px 8px;
           border-radius: 99px; font-weight: 600; }
  .badge.accept { background: #14532d; color: var(--accent); }
  .badge.review { background: #431407; color: var(--warn); }
  .badge.reject { background: #450a0a; color: var(--danger); }

  .filter-section { display: flex; flex-direction: column; gap: 8px; }
  .filter-label { font-size: 11px; color: var(--muted); font-family: var(--mono);
                  text-transform: uppercase; letter-spacing: 1px; }
  select, .filter-btn { background: var(--card); border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; padding: 7px 10px;
    font-family: var(--sans); font-size: 13px; cursor: pointer; width: 100%; }
  select:focus { outline: none; border-color: var(--accent2); }
  .filter-btn { text-align: center; transition: all .2s; }
  .filter-btn:hover { background: var(--border); }

  /* ── Main ── */
  .main { padding: 32px 40px; display: flex; flex-direction: column; gap: 24px; }

  /* ── Header ── */
  .header { display: flex; align-items: center; justify-content: space-between; }
  .header-title { font-family: var(--mono); font-size: 18px; font-weight: 600; }
  .header-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .nav-btns { display: flex; gap: 8px; }
  .nav-btn { background: var(--card); border: 1px solid var(--border);
             color: var(--text); border-radius: 8px; padding: 8px 18px;
             font-family: var(--mono); font-size: 12px; cursor: pointer;
             transition: all .2s; }
  .nav-btn:hover { border-color: var(--accent2); color: var(--accent2); }
  .nav-btn:disabled { opacity: .3; cursor: not-allowed; }

  /* ── Record card ── */
  .record-card { background: var(--card); border: 1px solid var(--border);
                 border-radius: 12px; overflow: hidden; }
  .record-header { background: var(--surface); border-bottom: 1px solid var(--border);
                   padding: 12px 20px; display: flex; align-items: center;
                   justify-content: space-between; }
  .record-index { font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .field-tag { font-family: var(--mono); font-size: 11px; font-weight: 600;
               padding: 3px 10px; border-radius: 99px; text-transform: uppercase;
               letter-spacing: 1px; }
  .field-tag.id       { background: #1e3a5f; color: var(--accent2); }
  .field-tag.name     { background: #2d1b69; color: #a78bfa; }
  .field-tag.birth    { background: #14532d; color: var(--accent); }
  .field-tag.origin   { background: #431407; color: var(--warn); }
  .field-tag.address  { background: #450a0a; color: var(--danger); }
  .field-tag.title    { background: #1c1917; color: #a8a29e; }

  .record-body { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
  .record-section { padding: 20px; }
  .record-section + .record-section { border-left: 1px solid var(--border); }

  .section-label { font-size: 10px; font-family: var(--mono); color: var(--muted);
                   text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }

  /* Crop image */
  .crop-wrap { background: #000; border-radius: 8px; overflow: hidden;
               display: flex; align-items: center; justify-content: center;
               min-height: 80px; max-height: 200px; }
  .crop-img { max-width: 100%; max-height: 200px; object-fit: contain;
              image-rendering: -webkit-optimize-contrast; }
  .crop-missing { color: var(--muted); font-size: 12px; padding: 24px;
                  font-family: var(--mono); }

  /* OCR candidates */
  .candidate { background: var(--surface); border: 1px solid var(--border);
               border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
  .candidate-header { display: flex; justify-content: space-between;
                      align-items: center; margin-bottom: 6px; }
  .candidate-source { font-size: 10px; font-family: var(--mono); color: var(--muted);
                      text-transform: uppercase; letter-spacing: 1px; }
  .conf-bar-wrap { display: flex; align-items: center; gap: 8px; }
  .conf-bar-bg { width: 80px; height: 4px; background: var(--border);
                 border-radius: 2px; overflow: hidden; }
  .conf-bar-fill { height: 100%; border-radius: 2px; }
  .conf-val { font-family: var(--mono); font-size: 10px; }
  .candidate-text { font-family: var(--mono); font-size: 14px; color: var(--text);
                    word-break: break-all; }
  .candidate-text.best { color: var(--accent); }
  .candidate-text.empty { color: var(--muted); font-style: italic; }

  /* Input area */
  .input-group { display: flex; flex-direction: column; gap: 8px; }
  .gt-label { font-size: 11px; font-family: var(--mono); color: var(--accent);
              text-transform: uppercase; letter-spacing: 2px; }
  .gt-input { background: var(--bg); border: 2px solid var(--accent);
              color: var(--accent); border-radius: 8px; padding: 12px 16px;
              font-family: var(--mono); font-size: 16px; width: 100%;
              transition: all .2s; letter-spacing: 1px; }
  .gt-input:focus { outline: none; border-color: #86efac; box-shadow: 0 0 0 3px #14532d; }
  .gt-input.has-value { border-color: var(--accent); }
  .gt-input.empty-warn { border-color: var(--warn); color: var(--text); }

  .hint-btns { display: flex; gap: 6px; flex-wrap: wrap; }
  .hint-btn { background: var(--surface); border: 1px solid var(--border);
              color: var(--text); border-radius: 6px; padding: 5px 10px;
              font-family: var(--mono); font-size: 12px; cursor: pointer;
              transition: all .15s; }
  .hint-btn:hover { border-color: var(--accent2); color: var(--accent2); }

  /* Action buttons */
  .actions { display: flex; gap: 10px; padding: 16px 20px;
             border-top: 1px solid var(--border); background: var(--surface); }
  .btn { border: none; border-radius: 8px; padding: 10px 24px;
         font-family: var(--mono); font-size: 13px; font-weight: 600;
         cursor: pointer; transition: all .2s; letter-spacing: .5px; }
  .btn-accept  { background: #14532d; color: var(--accent); }
  .btn-accept:hover  { background: #166534; }
  .btn-skip    { background: var(--card); color: var(--muted);
                 border: 1px solid var(--border); }
  .btn-skip:hover    { color: var(--text); border-color: var(--muted); }
  .btn-save    { background: var(--accent2); color: #0c1a2e; margin-left: auto; }
  .btn-save:hover    { background: #7dd3fc; }

  /* Meta info */
  .meta-row { display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px;
              color: var(--muted); font-family: var(--mono); }
  .meta-item span { color: var(--text); }

  /* Toast */
  .toast { position: fixed; bottom: 24px; right: 24px; background: var(--accent);
           color: #052e16; padding: 10px 20px; border-radius: 8px;
           font-family: var(--mono); font-weight: 600; font-size: 13px;
           opacity: 0; transform: translateY(10px);
           transition: all .3s; pointer-events: none; z-index: 999; }
  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.error { background: var(--danger); color: #450a0a; }

  /* Empty state */
  .empty-state { text-align: center; padding: 80px 20px; color: var(--muted); }
  .empty-state h2 { font-family: var(--mono); color: var(--accent); margin-bottom: 8px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="layout">

<!-- ── Sidebar ── -->
<aside class="sidebar">
  <div class="sidebar-title">⬡ OCR Review</div>

  <div class="stat-block">
    <div class="stat-label">Tiến độ</div>
    <div class="stat-value green" id="stat-done">0</div>
    <div style="font-size:12px;color:var(--muted);margin:4px 0 8px">/ <span id="stat-total">0</span> records</div>
    <div class="progress-wrap">
      <div class="progress-bar" id="progress-bar" style="width:0%"></div>
    </div>
  </div>

  <div class="stat-block">
    <div class="stat-label">Còn lại</div>
    <div class="stat-value orange" id="stat-remaining">0</div>
  </div>

  <div class="stat-block">
    <div class="stat-label">Phân loại</div>
    <div id="bucket-list"></div>
  </div>

  <div class="filter-section">
    <div class="filter-label">Lọc theo field</div>
    <select id="filter-field" onchange="applyFilter()">
      <option value="all">Tất cả</option>
      <option value="id">id_number</option>
      <option value="name">full_name</option>
      <option value="birth">date_of_birth</option>
      <option value="origin">place_of_origin</option>
      <option value="address">place_of_residence</option>
    </select>
  </div>

  <div class="filter-section">
    <div class="filter-label">Lọc theo trạng thái</div>
    <select id="filter-status" onchange="applyFilter()">
      <option value="all">Tất cả</option>
      <option value="pending">Chưa review (ground_truth trống)</option>
      <option value="done">Đã review</option>
      <option value="reject">Bucket: reject</option>
      <option value="review">Bucket: review</option>
    </select>
  </div>

  <button class="filter-btn" onclick="saveAll()">💾 Lưu tất cả</button>
  <button class="filter-btn" onclick="exportDone()">📤 Export reviewed_final.jsonl</button>
</aside>

<!-- ── Main ── -->
<main class="main">
  <div class="header">
    <div>
      <div class="header-title">CCCD Pseudo-Label Review</div>
      <div class="header-sub" id="header-sub">Đang tải...</div>
    </div>
    <div class="nav-btns">
      <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">← Trước</button>
      <button class="nav-btn" id="btn-next" onclick="navigate(1)">Tiếp →</button>
    </div>
  </div>

  <div id="record-area"></div>
</main>
</div>

<div class="toast" id="toast"></div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let ALL_RECORDS = [];
let FILTERED    = [];
let CUR_IDX     = 0;

// ── Boot ──────────────────────────────────────────────────────────────────
async function boot() {
  const res  = await fetch('/api/records');
  ALL_RECORDS = await res.json();
  applyFilter();
  updateStats();
}

// ── Filter ──────────────────────────────────────────────────────────────────
function applyFilter() {
  const fField  = document.getElementById('filter-field').value;
  const fStatus = document.getElementById('filter-status').value;
  FILTERED = ALL_RECORDS.filter(r => {
    const fieldOk = fField === 'all' || r.class === fField;
    const gt = (r.ground_truth_text || '').trim();
    let statusOk = true;
    if (fStatus === 'pending') statusOk = !gt;
    if (fStatus === 'done')    statusOk = !!gt;
    if (fStatus === 'reject')  statusOk = r.review_bucket === 'reject';
    if (fStatus === 'review')  statusOk = r.review_bucket === 'review';
    return fieldOk && statusOk;
  });
  CUR_IDX = 0;
  render();
  updateStats();
}

// ── Navigate ──────────────────────────────────────────────────────────────
function navigate(dir) {
  saveCurrentInput();
  CUR_IDX = Math.max(0, Math.min(FILTERED.length - 1, CUR_IDX + dir));
  render();
}

function saveCurrentInput() {
  if (!FILTERED[CUR_IDX]) return;
  const inp = document.getElementById('gt-input');
  if (!inp) return;
  const val = inp.value.trim();
  const rec = FILTERED[CUR_IDX];
  rec.ground_truth_text = val;
  // sync back to ALL_RECORDS
  const orig = ALL_RECORDS.find(r => r.ann_id === rec.ann_id && r.crop_path === rec.crop_path);
  if (orig) orig.ground_truth_text = val;
}

// ── Render ──────────────────────────────────────────────────────────────────
function render() {
  const area = document.getElementById('record-area');
  if (!FILTERED.length) {
    area.innerHTML = `<div class="empty-state"><h2>Không có record nào</h2><p>Thử thay đổi bộ lọc</p></div>`;
    document.getElementById('header-sub').textContent = '0 records';
    return;
  }

  const rec = FILTERED[CUR_IDX];
  document.getElementById('header-sub').textContent =
    `Record ${CUR_IDX + 1} / ${FILTERED.length}  •  ${rec.source_image || ''}`;
  document.getElementById('btn-prev').disabled = CUR_IDX === 0;
  document.getElementById('btn-next').disabled = CUR_IDX === FILTERED.length - 1;

  const gt = rec.ground_truth_text || '';
  const bestText = rec.best_text || '';
  const confColor = conf => conf >= 0.9 ? '#4ade80' : conf >= 0.5 ? '#fb923c' : '#f87171';

  // Build candidates HTML
  let candidatesHtml = '';
  const cands = rec.candidates || {};
  for (const [src, c] of Object.entries(cands)) {
    const isBest = src === rec.best_source || (rec.best_source === 'vietocr' && src === 'vietocr')
                   || (rec.best_source === 'confidence_tiebreak' && src === 'vietocr');
    const txt = c.text || '';
    candidatesHtml += `
    <div class="candidate">
      <div class="candidate-header">
        <span class="candidate-source">${src}</span>
        <div class="conf-bar-wrap">
          <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:${(c.confidence*100).toFixed(0)}%;background:${confColor(c.confidence)}"></div>
          </div>
          <span class="conf-val" style="color:${confColor(c.confidence)}">${(c.confidence*100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="candidate-text ${isBest?'best':''} ${!txt?'empty':''}">${txt || '(trống)'}</div>
    </div>`;
  }

  // Hint buttons from candidates
  const hints = [...new Set(Object.values(cands).map(c => c.text).filter(Boolean))];
  const hintBtns = hints.map(h =>
    `<button class="hint-btn" onclick="useHint(this)" data-val="${h.replace(/"/g,'&quot;')}">${h}</button>`
  ).join('');

  const bucketClass = {'accept':'accept','review':'review','reject':'reject'}[rec.review_bucket] || 'review';

  area.innerHTML = `
  <div class="record-card">
    <div class="record-header">
      <div class="record-index">#${rec.ann_id ?? CUR_IDX} &nbsp;•&nbsp; ${rec.split || ''}</div>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="badge ${bucketClass}">${rec.review_bucket || '?'}</span>
        <span class="field-tag ${rec.class || ''}">${rec.field_name || rec.class || ''}</span>
      </div>
    </div>

    <div class="record-body">
      <!-- Left: image + meta -->
      <div class="record-section">
        <div class="section-label">Ảnh crop</div>
        <div class="crop-wrap" id="crop-wrap">
          <span class="crop-missing">Đang tải ảnh...</span>
        </div>
        <div class="meta-row" style="margin-top:12px">
          <div class="meta-item">class: <span>${rec.class || '?'}</span></div>
          <div class="meta-item">conf: <span style="color:${confColor(rec.best_conf)}">${((rec.best_conf||0)*100).toFixed(1)}%</span></div>
        </div>
        <div style="margin-top:8px;font-size:10px;color:var(--muted);font-family:var(--mono);word-break:break-all">${rec.crop_path || ''}</div>
      </div>

      <!-- Right: OCR + input -->
      <div class="record-section">
        <div class="section-label">OCR Candidates</div>
        ${candidatesHtml}

        <div class="input-group" style="margin-top:16px">
          <div class="gt-label">✎ Ground Truth</div>
          ${hintBtns.length ? `<div class="hint-btns">${hintBtns}</div>` : ''}
          <input id="gt-input"
                 class="gt-input ${gt ? 'has-value' : 'empty-warn'}"
                 type="text"
                 value="${gt.replace(/"/g,'&quot;')}"
                 placeholder="Nhập text đúng từ ảnh..."
                 onkeydown="handleKey(event)"
                 oninput="onInputChange(this)">
          <div style="font-size:11px;color:var(--muted);font-family:var(--mono)">
            Enter = Accept &amp; next &nbsp;│&nbsp; Tab = Skip
          </div>
        </div>
      </div>
    </div>

    <div class="actions">
      <button class="btn btn-accept" onclick="accept()">✓ Accept &amp; Next</button>
      <button class="btn btn-skip"   onclick="skip()">→ Skip</button>
      <button class="btn btn-save"   onclick="saveAll()">💾 Lưu</button>
    </div>
  </div>`;

  // Load image async
  loadImage(rec.crop_path);

  // Focus input
  setTimeout(() => {
    const inp = document.getElementById('gt-input');
    if (inp) { inp.focus(); inp.select(); }
  }, 50);
}

async function loadImage(cropPath) {
  if (!cropPath) return;
  try {
    const res = await fetch('/api/image?path=' + encodeURIComponent(cropPath));
    if (!res.ok) return;
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const wrap = document.getElementById('crop-wrap');
    if (wrap) wrap.innerHTML = `<img class="crop-img" src="${url}" alt="crop">`;
  } catch(e) {}
}

// ── Interactions ──────────────────────────────────────────────────────────
function onInputChange(inp) {
  inp.className = 'gt-input ' + (inp.value.trim() ? 'has-value' : 'empty-warn');
}

function useHint(btn) {
  const inp = document.getElementById('gt-input');
  inp.value = btn.dataset.val;
  onInputChange(inp);
  inp.focus();
}

function handleKey(e) {
  if (e.key === 'Enter')  { e.preventDefault(); accept(); }
  if (e.key === 'Tab')    { e.preventDefault(); skip(); }
}

function accept() {
  const inp = document.getElementById('gt-input');
  if (!inp) return;
  const val = inp.value.trim();
  if (!val) { showToast('Hãy nhập ground truth trước!', true); return; }
  const rec = FILTERED[CUR_IDX];
  rec.ground_truth_text = val;
  const orig = ALL_RECORDS.find(r => r.crop_path === rec.crop_path);
  if (orig) orig.ground_truth_text = val;
  updateStats();
  navigate(1);
}

function skip() { navigate(1); }

// ── Save ──────────────────────────────────────────────────────────────────
async function saveAll() {
  saveCurrentInput();
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(ALL_RECORDS)
  });
  const data = await res.json();
  showToast(data.ok ? `✓ Đã lưu ${data.saved} records` : '✗ Lỗi khi lưu', !data.ok);
}

async function exportDone() {
  saveCurrentInput();
  const done = ALL_RECORDS.filter(r => (r.ground_truth_text||'').trim());
  const res = await fetch('/api/export', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(done)
  });
  const data = await res.json();
  showToast(data.ok ? `✓ Exported ${data.saved} records → reviewed_final.jsonl` : '✗ Lỗi', !data.ok);
}

// ── Stats ──────────────────────────────────────────────────────────────────
function updateStats() {
  const total     = ALL_RECORDS.length;
  const done      = ALL_RECORDS.filter(r => (r.ground_truth_text||'').trim()).length;
  const remaining = total - done;
  document.getElementById('stat-total').textContent     = total;
  document.getElementById('stat-done').textContent      = done;
  document.getElementById('stat-remaining').textContent = remaining;
  document.getElementById('progress-bar').style.width   = (total ? (done/total*100).toFixed(1) : 0) + '%';

  // bucket counts
  const buckets = {};
  ALL_RECORDS.forEach(r => { const b = r.review_bucket||'?'; buckets[b]=(buckets[b]||0)+1; });
  const bl = document.getElementById('bucket-list');
  bl.innerHTML = Object.entries(buckets).map(([b,n]) =>
    `<div class="bucket-row"><span>${b}</span><span class="badge ${b}">${n}</span></div>`
  ).join('');
}

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(msg, isError=false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => { t.className = 'toast'; }, 2500);
}

// ── Keyboard shortcut ──────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.target.id === 'gt-input') return;
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
  if ((e.ctrlKey||e.metaKey) && e.key === 's') { e.preventDefault(); saveAll(); }
});

boot();
</script>
</body>
</html>"""

# ── HTTP Server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    records      = []
    input_path   = ""
    output_path  = ""
    export_path  = ""

    def log_message(self, fmt, *args):
        pass  # tắt log mặc định

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_html(HTML_TEMPLATE)

        elif parsed.path == "/api/records":
            self.send_json(Handler.records)

        elif parsed.path == "/api/image":
            qs   = parse_qs(parsed.query)
            path = qs.get("path", [""])[0]
            self._serve_image(path)

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)
        data    = json.loads(body)
        parsed  = urlparse(self.path)

        if parsed.path == "/api/save":
            Handler.records = data
            try:
                save_records(data, Handler.output_path)
                self.send_json({"ok": True, "saved": len(data)})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)})

        elif parsed.path == "/api/export":
            try:
                save_records(data, Handler.export_path)
                self.send_json({"ok": True, "saved": len(data)})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)})

    # ── helpers ──────────────────────────────────────────────────────────────

    def _serve_image(self, path):
        if not path or not os.path.exists(path):
            self.send_response(404); self.end_headers(); return
        ext  = Path(path).suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")
        try:
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_response(500); self.end_headers()

    def send_html(self, html: str):
        b = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(b))
        self.end_headers()
        self.wfile.write(b)

    def send_json(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(b))
        self.end_headers()
        self.wfile.write(b)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OCR Pseudo-Label Review Tool")
    parser.add_argument("--input",  default="data/processed/ocr/reviewed.jsonl",
                        help="Path tới file reviewed.jsonl")
    parser.add_argument("--output", default="",
                        help="Path lưu (mặc định: ghi đè input)")
    parser.add_argument("--export", default="data/processed/ocr/reviewed_final.jsonl",
                        help="Path export file đã review xong")
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or args.input
    export_path = args.export

    if not os.path.exists(input_path):
        print(f"[ERROR] Không tìm thấy file: {input_path}")
        return

    records = load_records(input_path)
    # Chuyển đổi format pseudo_labels.jsonl sang format tool cần
    for r in records:
        if "candidates" not in r:
            r["candidates"] = {}
            if r.get("text_vietocr") is not None:
                r["candidates"]["vietocr"] = {
                    "text": r.get("text_vietocr", ""),
                    "confidence": r.get("conf_vietocr", 0.0),
                }
            if r.get("text_easyocr") is not None:
                r["candidates"]["easyocr"] = {
                    "text": r.get("text_easyocr", ""),
                    "confidence": r.get("conf_easyocr", 0.0),
                }
        if "best_source" not in r:
            r["best_source"] = "vietocr"
        if "field_name" not in r:
            r["field_name"] = r.get("class", "")
        if "ann_id" not in r:
            r["ann_id"] = records.index(r)
    print(f"[INFO] Loaded {len(records)} records từ {input_path}")

    stats = compute_stats(records)
    print(f"[INFO] Đã có ground_truth: {stats['done']}/{stats['total']} "
          f"({stats['done']/max(stats['total'],1)*100:.1f}%)")
    print(f"[INFO] Bucket breakdown: {stats['buckets']}")

    Handler.records     = records
    Handler.input_path  = input_path
    Handler.output_path = output_path
    Handler.export_path = export_path

    server = HTTPServer(("localhost", args.port), Handler)
    url    = f"http://localhost:{args.port}"
    print(f"\n[READY] Review tool chạy tại: {url}")
    print(f"[INFO]  Lưu vào: {output_path}")
    print(f"[INFO]  Export  : {export_path}")
    print(f"[INFO]  Nhấn Ctrl+C để dừng\n")

    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[STOP] Đã dừng server.")
        # Auto-save khi thoát
        save_records(Handler.records, output_path)
        print(f"[SAVE] Đã lưu {len(Handler.records)} records → {output_path}")


if __name__ == "__main__":
    main()
