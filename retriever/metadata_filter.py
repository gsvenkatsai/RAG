# results = [
#     {
#         "text": "Python decorators wrap functions",
#         "topic": "python",
#         "source": "python_notes.pdf"
#     },
#     {
#         "text": "Django ORM handles models",
#         "topic": "django",
#         "source": "django_notes.pdf"
#     }
# ]

# filtered_results = filter_by_metadata(
#     results,
#     required_topic="python"
# )

# [
#     {
#         "text": "Python decorators wrap functions",
#         "topic": "python",
#         "source": "python_notes.pdf"
#     }
# ]

def filter_by_metadata(results, required_topic=None, required_source=None):
    filtered_results = []

    for item in results:
        topic_match = True
        source_match = True

        if required_topic:
            topic_match = item.get("topic") == required_topic

        if required_source:
            source_match = item.get("source") == required_source

        if topic_match and source_match:
            filtered_results.append(item)

    return filtered_results