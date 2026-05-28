/**
 * toVideo function transforms a raw match object into a normalised video object used by the user interface
 * 
 * Function standardises field names and derives display-friendly values for rendering
 * 
 * @param {Object} m - Raw match object 
 * @returns {Object} Normalised video object
 */
export function toVideo(m) {
  return {
    id: m.id,
    source: 'library',
    match: m.title,
    tournament: [m.tournament, m.year, m.round].filter(Boolean).join(' '),
    duration: '—',
    strokes: m.strokes,
    annotated: true,
    youtubeId: m.youtubeId,
    url: m.url,
    fps: m.fps,
    sets: m.sets,
    year: m.year,
    round: m.round,
    strokeTimes: m.strokeTimes || [],
  };
}