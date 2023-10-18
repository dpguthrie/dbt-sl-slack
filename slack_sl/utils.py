# stdlib
import base64


def to_arrow_table(byte_string: str, to_pandas: bool = True):
    import pyarrow as pa

    with pa.ipc.open_stream(base64.b64decode(byte_string)) as reader:
        arrow_table = pa.Table.from_batches(reader, reader.schema)

    if to_pandas:
        return arrow_table.to_pandas()

    return arrow_table


def post_ephemeral_message(url: str, msg: str = None) -> None:
    import requests

    payload = {"text": msg}
    requests.post(url, json=payload)
