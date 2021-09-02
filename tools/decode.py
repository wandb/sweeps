import json
import base64
import sys

fname = sys.argv[1]


def convert(string):
    return json.loads(base64.b64decode(string))


def walk_dict(d):
    rewalk = False
    for k, v in d.items():
        if isinstance(v, dict):
            walk_dict(v)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    walk_dict(item)
                elif isinstance(item, str):
                    try:
                        v[i] = convert(item)
                    except Exception:
                        try:
                            v[i] = json.loads(item)
                        except Exception:
                            v[i] = item
                        else:
                            rewalk = True
                    else:
                        rewalk = True
        elif isinstance(v, str):
            try:
                d[k] = convert(v)
            except Exception:
                try:
                    d[k] = json.loads(v)
                except Exception:
                    d[k] = v
                else:
                    rewalk = True
            else:
                rewalk = True
    if rewalk:
        walk_dict(d)


with open(fname, "r") as f:
    d = json.load(f)

walk_dict(d)
print(json.dumps(d, indent=2))
