import sys
import json
import sweeps
import requests

sys.path.insert(0, "/Users/danielgoldstein/PycharmProjects")

fname = sys.argv[1]
with open(fname, "r") as f:
    data = json.load(f)

runs = [sweeps.SweepRun(**run) for run in data["jsonPayload"]["data"]["runs"]]
config = data["jsonPayload"]["data"]["config"]
requestId = data["jsonPayload"]["data"]["requestId"]


def send_anaconda(version=1, method="search"):
    rr = [r.dict(by_alias=True) for r in runs]
    for r in rr:
        r["history"] = r["sampledHistory"]
        del r["sampledHistory"]

    # return requests.post(f"http://localhost:{8081 + (version == 1)}/{method}", json={'requestId': requestId, 'config': config, 'runs': rr})
    return requests.post(
        f"http://anaconda{version if version > 1 else ''}.test/{method}",
        json={
            "requestId": requestId,
            "config": config,
            "runs": [r.dict(by_alias=True) for r in runs],
        },
    )
