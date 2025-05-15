async function  requestInfo() {

    const url = 'http://localhost:5000/info'

    try {
        const response = await fetch( url, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
            }, 
        } )
        // console.log( response.status, response.statusText )
        const result = await response.json();

        if ( response.status !== 200 ) {
            return {
                isError: true, 
                statusCode: response.status,
                statusText: response.statusText,
                message: result.detail
            }
        }
        return result;

    } catch ( error ) {
        return {
            isError: true, 
            message: error
        }
    }
}

export { requestInfo }
