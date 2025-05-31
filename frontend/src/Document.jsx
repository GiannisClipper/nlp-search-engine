import './styles/Document.css'

function Document( { document } ) {

    const { id, title, authors, published, summarized, catg_ids, judge } = document

    const judgeClass = judge===true?'yes':judge===false?'no':''

    return (
        <div className={'document '+judgeClass}>
            <div className='supertitle'>
                <div className='left'>{published} | {authors.join(', ')}</div>
                <div className='right'>[{catg_ids.join(' ')}]</div>
            </div>
            <div className='title'>{title}</div>
            <div className='summarized'>{summarized}</div>
            <a className='id' href={id}>{id}</a>
        </div>
    )
}

export default Document
