import { useState } from 'react'
// import './App.css'

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

function SearchOption( { onRequest, setOnRequest } ) {
    return (
        <div className='search option'>
            <div className='params'>
                <div className='top'>
                    <input 
                        className='query' 
                        placeholder='Please enter your query here...' 
                        disabled={onRequest?true:false}
                    />
                    {
                    ! 
                    onRequest
                    ?
                    <button 
                        className='go' 
                        onClick={() => setOnRequest( 'onRequest' )}
                        disabled={onRequest?true:false}
                    >
                        [Go]
                    </button>
                    :
                    <div 
                        className='loader'
                    >
                        ...
                    </div>
                    }
                </div>
                <div className='bottom'>
                    <input 
                        className='names' 
                        placeholder='Optional autor(s) [name1,name2...]' 
                        disabled={onRequest?true:false}
                    />
                    <input 
                        className='period' 
                        placeholder='Optional published period [yyyy-mm-dd,yyyy-mm-dd]' 
                        disabled={onRequest?true:false}
                    />
                </div>
            </div>
        </div>
    )
}

function HistoryOption() {
    return <div>'History...'</div>
}

export default App
