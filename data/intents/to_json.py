import yaml
import json

in_file = 'data/intents/nlu.yml'
out_file = 'data/intents/nlu.json'

with open(in_file) as f:
    data = yaml.safe_load(f)

res = []
for _data in data['nlu']:
    intent, examples = _data['intent'], _data['examples']
    examples = [line.removeprefix('- ') for line in examples.splitlines()]
    for example in examples:
        res.append(dict(text=example, intent=intent))

with open(out_file, 'w') as f:
    json.dump(res, f, indent=2)