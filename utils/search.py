def get_client(vectorstore):
    if vectorstore == 'elasticsearch':
        from elasticsearch import Elasticsearch

        # Create a client instance
        client = Elasticsearch("http://localhost:9200")  # Adjust the host URL if needed
    else:
        raise NotImplementedError(f"Not implemented for {vectorstore}")
    return client


def search(client, index_name, query):
    search_query = {
        "query": {
            "query_string": {
                "query": query
            }
        }
    }
    response = client.search(index=index_name, body=search_query)
    return response