import json
from dynamorator import DynamoDBStore

key = "e611a1efe8800663"
store = DynamoDBStore(table_name="audible-toolkit-llm", silent=True)

result = store.get(key)
if result is None:
    print(f"No cache entry found for key: {key}")
else:
    print(json.dumps(result, indent=2, default=str))
