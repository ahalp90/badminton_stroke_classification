import { useState, useEffect } from 'react'

export default function Analysis() {
    const [models, setModels] = useState([])
    const [status, setStatus] = useState('')
    
    // Run once on page load, to get available models
    useEffect(() => {
        fetch("http://127.0.0.1:8000/api/models")
              .then((response) => response.json())
              .then((json) => setModels(json.models))
              .catch((error) => console.error('Error fetching data: ', error))
    }, [])

    useEffect(() => {
        const interval = setInterval(() => {
            fetch("http://127.0.0.1:8000/api/status/123")
              .then((response) => response.json())
              .then((json) => {
                setStatus(json.status)
                console.log(json.status)
                if (json.status === "complete") {
                    clearInterval(interval) // stop polling
                }
              })
              .catch((error) => console.error('Error fetching data: ', error))
        }, 2000)
        return () => clearInterval(interval)
    }, [])

    return (
        <>
          <div>Analysis</div>
          <div>Select a model:</div>
          <ul>
            {models.map((model, index) => (
                <li key={index}>{model}</li>
            ))}
          </ul>
          <div>Stroke classification is: {status}</div>
        </>

    )
}