import { useState, useEffect } from 'react'

export default function Results() {
    const [results, setResults] = useState(null)
    
    // Run once on page load, to get available results
    useEffect(() => {
        fetch("http://127.0.0.1:8000/api/results/123")
              .then((response) => response.json())
              .then((json) => setResults(json))
              .catch((error) => console.error('Error fetching data: ', error))
    }, [])

    if (!results) return <div>Loading...</div>
    
    return (
        <>
          <h2>Results</h2>
          <h3>Rally Summary</h3>
          <p>Total Strokes: {results.rally_summary.total_strokes}</p>
          <p>Rally Length: {results.rally_summary.rally_length_seconds} seconds</p>

          <h3>Strokes</h3>
          {results.strokes.map((stroke, index) => (
            <div key={index}>
                <p>Timestamp: {stroke.timestamp_sec}s</p>
                <p>Type: {stroke.stroke_type}</p>
                <p>Confidence: {stroke.confidence}</p>
            </div>
        ))}
        </>
    )
}