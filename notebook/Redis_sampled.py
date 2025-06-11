from redis.commands.search.suggestion import Suggestion

with redis_client.pipeline(transaction=False) as pipe:
    for key, val in most_common_keywords.items():
        total  = 0
        pipe.ft().sugadd(
            'top_action_keywords',
            Suggestion(key, val),
            increment=True,
        )
        total += 1
    pipe.execute()

print(f"Inserted/updated {total:,} keywords into 'top_action_keywords'")
