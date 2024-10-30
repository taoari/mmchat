
## Collections

```bash
# intents
python ./data/intents/to_json.py
python ./utils/vectorstore2.py -vs elasticsearch -c intents -f ./data/intents/nlu.json

# workplace
python ./utils/vectorstore2.py -vs elasticsearch -c workplace -f ./data/workplace/data.json
```

### Advanced Embeddings

```bash
python ./utils/vectorstore2.py -vs elasticsearch -c intents_adv -f ./data/intents/nlu.json --embeddings "http://localhost:8081;Alibaba-NLP/gte-large-en-v1.5"

python ./utils/vectorstore2.py -vs elasticsearch -c workplace_adv -f ./data/workplace/data.json --embeddings "http://localhost:8081;Alibaba-NLP/gte-large-en-v1.5"
```