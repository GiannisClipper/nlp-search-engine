import { useEffect, useState } from 'react'
import './styles/SearchOption.css'
import { requestSearch, requestJudge } from './helpers/requests'
import Document from './Document'

function SearchOption( { onRequest, setOnRequest } ) {

    const [ query, setQuery ] = useState( '' )
    const [ authors, setAuthors ] = useState( '' )
    const [ published, setPublished ] = useState( '' )
    const [ documents, setDocuments ] = useState( [] )

    useEffect( () => {
        if ( onRequest === 'go search' ) {
            const request = async () => {
                const result = await requestSearch( { query, authors, published } )
                if ( result.error ) {
                    alert( result.error.message )
                    setOnRequest( null )
                } else {
                    console.log( result )
                    setDocuments( result.data )
                    setOnRequest( null )
                }
            }
            request()
            setOnRequest( 'waiting' )
        }
        if ( onRequest === 'go judge' ) {
            const request = async () => {
                const idocs = documents.map( d => d[ 'idoc' ] )
                const result = await requestJudge( { query, idocs } )
                if ( result.error ) {
                    alert( result.error.message )
                    setOnRequest( null )
                } else {
                    console.log( result )
                    for ( let i=0; i<documents.length; i++ ) {
                        documents[ i ][ 'judge' ] = result.data[ i ][ documents[ i ][ 'idoc' ] ]
                    }
                    setDocuments( [ ...documents ] )
                    setOnRequest( null )
                }
            }
            request()
            setOnRequest( 'waiting' )
        }
    }, [onRequest] )

    console.log( 'documents:', documents )

    return (
        <div className='search-option'>
            <div className='params'>
                <div className='top'>
                    <input 
                        className='query' 
                        placeholder='Please enter your query...' 
                        disabled={onRequest?true:false}
                        onChange={ e => setQuery( e.target.value ) }
                    />
                    {
                    ! 
                    onRequest
                    ?
                    <>
                    <button 
                        className='search-tool' 
                        onClick={() => setOnRequest( 'go search' )}
                        disabled={onRequest?true:false}
                    >
                        [Find]
                    </button>
                    <button 
                        className='search-tool' 
                        onClick={() => setOnRequest( 'go judge' )}
                        disabled={onRequest?true:false}
                    >
                        [Eval]
                    </button>
                    </>
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
                        placeholder='Autor(s) like -> name1,name2...' 
                        disabled={onRequest?true:false}
                        onChange={ e => setAuthors( e.target.value ) }
                    />
                    <input 
                        className='period' 
                        placeholder='Published period like -> yyyy-mm-dd,yyyy-mm-dd' 
                        disabled={onRequest?true:false}
                        onChange={ e => setPublished( e.target.value ) }
                    />
                </div>
            </div>
            <div className='documents'>
                { documents.map( d => <Document document={d} /> ) }
            </div>
        </div>
    )
}

export default SearchOption
