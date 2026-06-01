  /**
   * useClipList drives the list panel that lists video clips meeting filter criteria
   *
   */
  import { useState, useEffect } from 'react';

  const CLIP_LIMIT = 50; // page size; full test split is ~4.2k clips, paged via offset

  export function useClipList({ modelId, split, errorsOnly, enabled = true }) {
    const [clips,  setClips]  = useState([]);
    const [total,  setTotal]  = useState(0);
    const [offset, setOffset] = useState(0);
    const [error,  setError]  = useState(null);
    
    // Reset to page 1 when filters change.
    useEffect(() => { setOffset(0); }, [modelId, split, errorsOnly]);
    
    // Pull the clip list whenever model / split / filter changes. Parent owns
    // model resolution; we just react to whatever modelId comes through.
    useEffect(() => {
        if (!modelId || !enabled) { setClips([]); setTotal(0); return; }
        let alive = true;
        setError(null);
        const params = new URLSearchParams({ limit: CLIP_LIMIT, offset });
        if (errorsOnly) params.set('errors_only', 'true');
        fetch(`/api/registry/${modelId}/splits/${split}/clips?${params}`)
          .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
          .then(data => {
            if (!alive) return
            const items = data.clips || [];
            setClips(items);
            setTotal(data.total ?? 0);
          })
        .catch(err => { if (alive) setError(err.message); });
        return () => { alive = false; };
    }, [modelId, split, errorsOnly, offset, enabled]);

    return { clips, total, offset, setOffset, limit: CLIP_LIMIT, error };
}