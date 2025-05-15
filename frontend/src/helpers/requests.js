async function  request( url, options ) {

    try {
        const response = await fetch( url, options )
        // console.log( response.status, response.statusText )
        const result = await response.json();

        if ( response.status !== 200 ) {
            return {
                error: { 
                    statusCode: response.status,
                    statusText: response.statusText,
                    message: result.detail
                }
            }
        }
        return {
            data: result
        };

    } catch ( error ) {
        return {
            error: {
                message: error
            }
        }
    }
}

async function  requestInfo() {

    const url = 'http://localhost:5000/info'

    const options = {
        method: "GET",
        headers: {
            "Content-Type": "application/json",
        },
    }

    return await request( url, options )
}

async function  requestSearch( { query, authors, published } ) {

    const url = 'http://localhost:5000/search'

    const options = {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        }, 
        body: JSON.stringify( { query, authors, published } )
    }

    return await request( url, options )
}

export { requestInfo, requestSearch }
