import './styles/Document.css'

function Document( { document } ) {

    const { id, title, authors, published, summary, catg_ids } = document

    return (
        <div className='document'>
            <div className='supertitle'>
                <div className='left'>{published} | {authors.join(', ')}</div>
                <div className='right'>[{catg_ids.join(' ')}]</div>
            </div>
            <div className='title'>{title}</div>
            <div className='summary'>{summary}</div>
            <a className='id' href={id}>{id}</a>
        </div>
    )
}

export default Document
