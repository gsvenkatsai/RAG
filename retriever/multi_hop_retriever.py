# query = "Who created FAISS and what is it used for?"
# [
#     "Who created FAISS",
#     "what is it used for?"
# ]

def split_multi_hop_query(query):
    separators = [" and ", ",", " then ", " also "]

    sub_queries = [query]

    for separator in separators:
        new_sub_queries = []

        for item in sub_queries:
            parts = item.split(separator)

            for part in parts:
                cleaned_part = part.strip()

                if cleaned_part:
                    new_sub_queries.append(cleaned_part)

        sub_queries = new_sub_queries

    return sub_queries
