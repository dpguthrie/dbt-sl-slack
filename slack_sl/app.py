# stdlib
import os
from typing import Dict

# third party
import asyncio
import modal
from fastapi import Request
from fastapi.responses import JSONResponse

# first party
from slack_sl.client import submit_request
from slack_sl.queries import GRAPHQL_QUERIES
from slack_sl.utils import post_ephemeral_message


stub = modal.Stub("semantic-layer-bot")


async def download_metrics_and_dimensions(local_files_only: bool = False) -> None:
    payload = {"query": GRAPHQL_QUERIES["metrics"]}
    response_data = await submit_request(payload)
    metrics_list = response_data.get("data", {}).get("metrics", [])
    if not metrics_list:
        raise ValueError("No metrics found.")

    metrics = [m["name"] for m in metrics_list]
    dimensions = list(set([d["name"] for m in metrics_list for d in m["dimensions"]]))

    return {
        "metrics": metrics,
        "dimensions": dimensions,
    }


image = (
    modal.Image.debian_slim()
    .pip_install(
        "langchain==0.0.306",
        "openai==0.28.1",
        "pydantic==2.4.2",
        "fastapi==0.103.2",
        "pyarrow==13.0.0",
        "pandas==2.1.1",
        "slack-sdk==3.23.0",
        "tabulate==0.9.0",
    )
    .run_function(
        download_metrics_and_dimensions,
        secrets=[modal.Secret.from_name("dbt-cloud-service-token")],
    )
)


@stub.function(
    container_idle_timeout=1200,
    image=image,
    secrets=[
        modal.Secret.from_name("dbt-cloud-service-token"),
        modal.Secret.from_name("openai-api-key"),
    ],
)
@modal.web_endpoint(method="POST")
async def question(request: Request):
    form_data = await request.form()
    dct = {k: v for k, v in form_data.items()}
    asyncio.create_task(process_question(dct))
    return JSONResponse(
        status_code=200, content={"text": "Working on your request now..."}
    )


async def process_question(dct: Dict):
    # third party
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from langchain.prompts.few_shot import FewShotPromptTemplate
    from langchain.schema.output_parser import OutputParserException
    from pydantic.v1.error_wrappers import ValidationError

    # first party
    from slack_sl.client import get_query_results
    from slack_sl.examples import EXAMPLES
    from slack_sl.prompt import EXAMPLE_PROMPT
    from slack_sl.schema import Query

    info = await download_metrics_and_dimensions(local_files_only=True)
    parser = PydanticOutputParser(pydantic_object=Query)
    prompt_example = PromptTemplate(
        template=EXAMPLE_PROMPT,
        input_variables=["metrics", "dimensions", "question", "result"],
    )
    prompt = FewShotPromptTemplate(
        examples=EXAMPLES,
        example_prompt=prompt_example,
        prefix="""Given a question involving a user's data, transform it into a structured object.
        {format_instructions}
        """,
        suffix="Metrics: {metrics}\nDimensions: {dimensions}\nQuestion: {question}\nResult:\n",
        input_variables=["metrics", "dimensions", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    try:
        llm = OpenAI(
            openai_api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-davinci-003",
            temperature=0,
        )

    except ValidationError as e:
        print("error:", e)
        return JSONResponse(
            content={"text": f"An error occured initializing an OpenAI account: {e}"}
        )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(
        metrics=info["metrics"],
        dimensions=info["dimensions"],
        question=dct["text"],
    )
    post_ephemeral_message(dct["response_url"], "🧠 Parsing output from LLM")
    try:
        query = parser.parse(output)
    except OutputParserException as e:
        return JSONResponse(
            content={"text": f"An error occured parsing your question: {e}"}
        )

    payload = {"query": query.gql, "variables": query.variables}
    results = await get_query_results(payload, dct["response_url"])
    response = await send_slack_message(results, dct["response_url"])
    print(response)


async def send_slack_message(results: Dict, response_url: str):
    # third party
    from slack_sdk.webhook import WebhookClient

    # first party
    from slack_sl.utils import to_arrow_table

    client = WebhookClient(response_url)
    table = to_arrow_table(results["arrowResult"])
    sql = results["sql"]

    # Convert the Pandas DataFrame to a Markdown table
    table_md = table.to_markdown(
        index=False,
        tablefmt="presto",
        intfmt=",",
    )

    table_response = client.send(text=f"```\n{table_md}\n```")
    sql_response = client.send(text=f"```\n{sql}\n```")
    print(table_response)
    return table_response


def validate_token(token: str):
    # validate token here
    pass
