import { useState, useEffect } from 'react'
import './styles/App.css'
import SearchOption from './SearchOption'
import HistoryOption from './HistoryOption'
import { requestInfo } from './helpers/requests'

function App() {

    const [ engineOption, setEngineOption ] = useState( 'unknown' )
    const [ option, setOption ] = useState( 'search' )
    const [ onRequest, setOnRequest ] = useState( null )
 
    useEffect( () => {
        const request = async () => {
            const result = await requestInfo()
            if ( result.error ) {
                alert( result.error.message )
                setEngineOption( 'unknown' )
            } else {
                setEngineOption( result.data.option )
            }
        }
        request()
    }, [] )

    return (
        <>
        <div className='header'>
            <div className='left'>
                <div className='title'>
                    arXiv search engine
                </div>
                <div className='subtitle'>
                    [ option:{engineOption} ]
                </div>
            </div>
            <div className='right'>
                {/* <div className={'search'+(option==='search'?' selected':'' )}>
                    <button 
                        onClick={() => setOption( 'search' )}
                        disabled={onRequest?true:false}
                    >
                        Search
                    </button>
                </div>
                <div 
                    className={'history'+(option==='history'?' selected':'' )}>
                    <button 
                        onClick={() => setOption( 'history' )}
                        disabled={onRequest?true:false}
                    >
                        History
                    </button>
                </div> */}
            </div>
        </div>
        <div className='main'>
            { 
            option==='search'
            ?
            <SearchOption 
                onRequest={onRequest}
                setOnRequest={setOnRequest}
            />
            : 
            option==='history'
            ?
            <HistoryOption />
            :
            null
            }
        </div>
        </>
    )
}

export default App
