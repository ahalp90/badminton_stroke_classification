import { Button } from '../components'

import SingleFileUploader from '../components/SingleFileUploader'

export default function Home() {

    return (
        <>
          <div>Home</div>
          <Button>Test Button</Button>
          <h1>File Upload:</h1>
          <SingleFileUploader/>
           
         </>
    )
}