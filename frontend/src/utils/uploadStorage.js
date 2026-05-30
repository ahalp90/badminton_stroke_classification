/* ─── Upload persistence (Gap 3: IndexedDB for blob bytes) ──────────
 *
 * Two layers:
 *   1. localStorage `bsc.uploads`  — small, JSON-serialisable metadata
 *      array. Survives refresh.
 *   2. IndexedDB `bsc / uploads`   — actual File / Blob bytes keyed by
 *      upload id. Survives refresh too, modulo the browser pruning
 *      storage under quota pressure.
 *   3. Module-level SESSION_FILES  — in-memory {file, objectURL} cache
 *      for the current page session, populated either by a fresh
 *      upload or by rehydrating from IndexedDB. Wiped on refresh; we
 *      repopulate it on module load via rehydrateSessionFromIDB().
 */
import { useState, useEffect } from 'react';

export const ACCEPTED_VIDEO_TYPES = 'video/mp4,video/quicktime,video/x-msvideo,video/*';

const UPLOADS_KEY = 'bsc.uploads';
const SESSION_FILES = new Map(); // id -> { file: File, objectURL: string }

const IDB_DB_NAME = 'bsc';
const IDB_STORE   = 'uploads';
let _idbPromise = null;
function idbOpen() {
  if (_idbPromise) return _idbPromise;
  _idbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'));
    const req = indexedDB.open(IDB_DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
  return _idbPromise;
}
async function idbPut(id, blob) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).put({ id, blob });
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
    tx.onabort    = () => reject(tx.error || new Error('IDB tx aborted'));
  });
}
async function idbGet(id) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readonly');
    const req = tx.objectStore(IDB_STORE).get(id);
    req.onsuccess = () => resolve(req.result?.blob ?? null);
    req.onerror   = () => reject(req.error);
  });
}
async function idbDelete(id) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    tx.objectStore(IDB_STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror    = () => reject(tx.error);
  });
}

export async function rehydrateSessionFromIDB() {
  if (typeof indexedDB === 'undefined') return;
  const entries = listStoredUploads();
  let restored = 0;
  for (const e of entries) {
    if (SESSION_FILES.has(e.id)) continue;
    try {
      const blob = await idbGet(e.id);
      if (!blob) continue;
      const file = new File([blob], e.filename || 'upload.mp4', { type: blob.type || 'video/mp4' });
      SESSION_FILES.set(e.id, { file, objectURL: URL.createObjectURL(file) });
      restored++;
    } catch { /* noop */ }
  }
  if (restored > 0) {
    window.dispatchEvent(new CustomEvent('bsc.uploads.changed'));
  }
}

// Kick off rehydration the moment the module loads so the first render
// of the My Uploads list mostly shows green status dots. The fetch is
// async; useStoredUploads listens for the 'bsc.uploads.changed' event
// dispatched once rehydration finishes.
if (typeof window !== 'undefined') {
  Promise.resolve().then(() => rehydrateSessionFromIDB().catch(() => { /* noop */ }));
}

export function listStoredUploads() {
  try { return JSON.parse(localStorage.getItem(UPLOADS_KEY) || '[]'); }
  catch { return []; }
}
function saveStoredUploads(arr) {
  localStorage.setItem(UPLOADS_KEY, JSON.stringify(arr));
  window.dispatchEvent(new CustomEvent('bsc.uploads.changed'));
}
export function recordUpload(file) {
  const id = `upload_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
  const entry = {
    id, filename: file.name, size: file.size,
    uploaded_at: new Date().toISOString(),
  };
  const arr = listStoredUploads();
  arr.unshift(entry);
  saveStoredUploads(arr);
  bindFileToUpload(id, file);
  // Persist the blob to IDB so it survives refresh. Best-effort: any
  // failure (quota, private mode) is logged but doesn't break the upload.
  idbPut(id, file).catch(err => console.warn('[bsc] IDB put failed for', id, err));
  return entry;
}
export function bindFileToUpload(id, file) {
  const prev = SESSION_FILES.get(id);
  if (prev?.objectURL) URL.revokeObjectURL(prev.objectURL);
  SESSION_FILES.set(id, { file, objectURL: URL.createObjectURL(file) });
  idbPut(id, file).catch(err => console.warn('[bsc] IDB rebind failed for', id, err));
}
export function deleteUpload(id) {
  const arr = listStoredUploads().filter(e => e.id !== id);
  saveStoredUploads(arr);
  const s = SESSION_FILES.get(id);
  if (s?.objectURL) URL.revokeObjectURL(s.objectURL);
  SESSION_FILES.delete(id);
  idbDelete(id).catch(err => console.warn('[bsc] IDB delete failed for', id, err));
}
export function getSessionFile(id) { return SESSION_FILES.get(id) ?? null; }
export function toUploadVideo(entry) {
  const s = SESSION_FILES.get(entry.id);
  return {
    id: entry.id,
    source: 'upload',
    match: entry.filename,
    tournament: `Your upload — ${new Date(entry.uploaded_at).toLocaleDateString('en-AU')}`,
    duration: '—',
    strokes: 0,
    annotated: false,
    file: s?.file ?? null,
    objectURL: s?.objectURL ?? null,
    filename: entry.filename,
    size: entry.size,
    uploadedAt: entry.uploaded_at,
    strokeTimes: [],
  };
}
export function fmtSize(bytes) {
  if (!bytes) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let v = bytes, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[u]}`;
}
export function useStoredUploads() {
  const [items, setItems] = useState(() => listStoredUploads());
  useEffect(() => {
    const refresh = () => setItems(listStoredUploads());
    window.addEventListener('bsc.uploads.changed', refresh);
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener('bsc.uploads.changed', refresh);
      window.removeEventListener('storage', refresh);
    };
  }, []);
  return items;
}
