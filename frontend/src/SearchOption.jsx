import { useState } from 'react'
import './styles/SearchOption.css'

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

export default SearchOption
