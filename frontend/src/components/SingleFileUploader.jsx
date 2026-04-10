// Adapted from: https://uploadcare.com/blog/how-to-upload-file-in-react/
import { useState } from 'react'

import style from './SingleFileUploader.module.css'

const SingleFileUploader = () => {
    const [file, setFile] = useState(null)
    
    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0])
        }
    }
        
    const handleUpload = async () => {
        if (file) {
            console.log('Uploading file...')
            
            const formData = new FormData()
            formData.append('file', file)

            try {
                const result = await fetch('http://127.0.0.1:8000/api/upload', {
                    method: 'POST',
                    body: formData,
                })
            
            const data = await result.json()

            console.log(data)
            } catch (error) {
                console.error(error)
            }
        }

    }
    
    return (
    <>
      <div className={style.input_group}>
        <input id="file" type="file" onChange={handleFileChange} />
      </div>
      {file && (
        <section>
            File details:
            <ul>
                <li>Name: {file.name}</li>
                <li>Type: {file.type}</li>
                <li>Size: {file.size} bytes</li>
            </ul>
        </section>
      )}

      {file && (
        <button 
        onClick={handleUpload}
        className={style.submit}
        >Upload a file</button>
        )}
    </>
  )
}

export default SingleFileUploader