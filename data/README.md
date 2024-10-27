
## Collections

```bash
# intents
python ./data/intents/to_json.py
python ./utils/vectorstore2.py -vs elasticsearch -c intents -f ./data/intents/nlu.json

# workplace
python ./utils/vectorstore2.py -vs elasticsearch -c workplace -f ./data/workplace/data.json
```