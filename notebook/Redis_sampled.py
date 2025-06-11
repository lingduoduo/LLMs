from redis.commands.search.suggestion import Suggestion

total = 0
BATCH_SIZE = 10_000

with redis_client.pipeline(transaction=False) as pipe:
    batch_count = 0
    for key, val in most_common_keywords.items():
        pipe.ft().sugadd(
            'top_action_keywords',
            Suggestion(key, float(val)),
            increment=True,
        )
        total += 1
        batch_count += 1

        if batch_count >= BATCH_SIZE:
            pipe.execute()
            batch_count = 0

    if batch_count > 0:
        pipe.execute()

print(f"Inserted/updated {total:,} keywords into 'top_action_keywords'")
