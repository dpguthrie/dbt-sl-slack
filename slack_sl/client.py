# stdlib
import aiohttp
import os
from typing import Dict

# third party
from fastapi.responses import JSONResponse

# first party
from slack_sl.queries import GRAPHQL_QUERIES
from slack_sl.utils import post_ephemeral_message


URL = 'https://semantic-layer.cloud.getdbt.com/api/graphql'


async def submit_request(payload):

    if 'variables' not in payload:
        payload['variables'] = {}
    payload['variables']['environmentId'] = 218762

    headers = {
        "Authorization": f"Bearer {os.environ['DBT_CLOUD_SERVICE_TOKEN']}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload, headers=headers) as response:
            return await response.json()


async def get_query_results(payload: Dict, response_url: str):
    post_ephemeral_message(response_url, '⏩ Submitting query...')
    json = await submit_request(payload)
    try:
        query_id = json["data"]["createQuery"]["queryId"]
    except TypeError as e:
        return JSONResponse(content={
            'text': f'An error occured generating a query: {e}'
        })

    post_ephemeral_message(response_url, '⏳ Waiting for results from query ...')
    while True:
        graphql_query = GRAPHQL_QUERIES["get_results"]
        results_payload = {"variables": {"queryId": query_id}, "query": graphql_query}
        json = await submit_request(results_payload)
        try:
            data = json["data"]["query"]
        except TypeError:
            error = json["errors"][0]["message"]
            return JSONResponse(content={
                'text': f'An error occured polling for results: {error}'
            })
        else:
            status = data["status"].lower()
            if status == "successful":
                break
            elif status == "failed":
                error = data["error"]
                return JSONResponse(content={
                    'text': f'An error occured submitting your query: {error}'
                })
            else:
                pass

    return data