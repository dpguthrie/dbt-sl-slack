# stdlib
import json

# third party
from langchain.llms import OpenAI


CHART_TYPE_FIELDS = {
    "line": ["x", "y", "color", "facet_row", "facet_col", "y2"],
    "bar": ["x", "y", "color", "orientation", "barmode", "y2"],
    "pie": ["values", "names"],
    "area": ["x", "y", "color", "y2"],
    "scatter": ["x", "y", "color", "size", "facet_col", "facet_row", "trendline"],
    "histogram": ["x", "nbins", "histfunc"],
}


CHART_TYPE_PROMPT = """
The following are the possible chart types supported by the code provided: area, bar, line, composed, scatter, and pie.\n
Given the user input: {data}, identify the chart type the user wants to display. Return just one word.
"""

ARGUMENTS_PROMPT = """
Given the following fields ({fields}) for a {chart_type} chart.  Pick out the keys from this list of
json objects {data} and map the keys to the appropriate fields (e.g. any date or time-based keys should
be mapped to the x field).  The data should be returned as a valid json object (for example
{{"x": "date", "y": "price"}})  All fields do not need to be used, the most important ones are x and y.
"""


def get_chart_type(llm: OpenAI, data: str):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    prompt = PromptTemplate.from_template(template=CHART_TYPE_PROMPT)
    formatted = prompt.format(data=data)
    chart_type = llm.predict(formatted).strip().lower()
    return chart_type


def map_columns_to_arguments(llm: OpenAI, data: str, chart_type: str):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate

    fields = CHART_TYPE_FIELDS[chart_type]
    prompt = PromptTemplate.from_template(template=ARGUMENTS_PROMPT)
    formatted = prompt.format(fields=fields, chart_type=chart_type, data=data)
    args_str = llm.predict(formatted).strip().lower()
    args_dict = json.loads(args_str)
    return {k: v.upper() for k, v in args_dict.items() if v is not None}
