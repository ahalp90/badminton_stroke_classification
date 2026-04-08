import { useState, useEffect } from 'react'

export default function Analysis() {
    const [models, setModels] = useState([])
    
    useEffect(() => {
        fetch("http://127.0.0.1:8000/api/models")
              .then((response) => response.json())
              .then((json) => setModels(json.models))
              .catch((error) => console.error('Error fetching data: ', error))
    }, [])

    return (
        <>
          <div>Analysis</div>
          <ul>
            {models.map((model, index) => (
                <li key={index}>{model}</li>
            ))}
          </ul>
        </>

    )
}