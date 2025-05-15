import { useState } from 'react'
import SearchOption from './SearchOption'
import HistoryOption from './HistoryOption'
import './styles/App.css'

function App() {

    const [ engineOption, setEngineOption ] = useState( 'unknown' )
    const [ option, setOption ] = useState( 'search' )
    const [ onRequest, setOnRequest ] = useState( null )

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
                <div className={'search'+(option==='search'?' selected':'' )}>
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
                </div>
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
