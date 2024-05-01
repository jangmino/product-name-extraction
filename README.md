# Developing a Model for Extracting Actual Product Names from Order Item Descriptions Using Generative Language Models



## 1. Build a JSONL file for OpenAI API

src/build_async_request_data.ipynb
- `local_data/order_item_description_samples.csv` contains 10,000 samples from a single day's worth of order item descriptions collected by Platform A. (Please understand that the full dataset cannot be disclosed due to the company's security policy.)
- This file samples order item descriptions, fills them into prompt templates, and generates a JSONL file that can be requested to the OpenAI API.
  
## 2. Request to OpenAI API

```bash
python src/openai/api_request_parallel_processor.py --requests_filepath=./local_data/api_requests_for_annotated_dataset.jsonl --save_filepath=./local_data/api_responses_for_annotated_dataset.jsonl --request_url=https://api.openai.com/v1/chat/completions --api_key=YOUR-API-KEY --max_requests_per_minute=1500 --max_tokens_per_minute=125000
```

## 3. Post-process the API responses

src/postprocess_api_responses.ipynb
- Post-process the responses received from the OpenAI API to generate an annotated dataset for order item descriptions.
- Although there are actually around 40,000 entries, a file consisting of a subset of about 3,000 entries is provided in `local_data/annotated_dataset_3K.pkl` (as it is a proprietary asset of the company.).

## 4. Fine-Tuning

After converting the annotated dataset into a labeled dataset in the NER format, perform fine-tuning.

KLUE/roberta-large

```bash
python src/trainer.py args/train-args-klue.json
```

FacebookAI/xlm-roberta-large

```bash
python src/trainer.py args/train-args-xlm-r.json
```
