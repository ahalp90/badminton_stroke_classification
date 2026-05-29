/**
 * useClipDetail drives the detail panel for a selected video clip
 * 
 */
import { useState, useEffect } from 'react';

export function useClipDetail ({ modelId, split, selectedStem }) {
    const [detail,  setDetail]  = useState(null);
    const [error,   setError]   = useState(null);
    const [loading, setLoading] = useState(false);
    
    // Pull the selected clip's per-clip JSON.
    useEffect(() => {
        if (!modelId || !selectedStem) { setDetail(null); return; }
        let alive = true;
        setLoading(true);
        setError(null);
        fetch(`/api/registry/${modelId}/splits/${split}/clips/${selectedStem}`)
          .then(response => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
          .then(data => { if (alive) { setDetail(data); setLoading(false); } })
          .catch(err => { if (alive) { setError(err.message); setLoading(false); } });

        return () => { alive = false; };
    }, [modelId, split, selectedStem]);

    return { detail, loading, error };
}