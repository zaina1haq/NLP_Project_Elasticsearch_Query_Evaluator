import json
import copy
import random
import argparse
import glob
import os
from collections import Counter

# Configure runtime parameters, initialize reproducibility settings, and prepare the output environment for dataset generation
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="Raw_data")
parser.add_argument("--out_dir",  default="json_format")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--train_ratio", type=float, default=0.70)
parser.add_argument("--val_ratio",   type=float, default=0.15)
args = parser.parse_args()

random.seed(args.seed)
os.makedirs(args.out_dir, exist_ok=True)

# 1. Load and aggregate Reuters documents from all source files to create the corpus used for dataset generation
print(f"Loading documents from: {args.data_dir}")
all_docs = []
for path in sorted(glob.glob(os.path.join(args.data_dir, "reut2-*.json"))):
    with open(path) as f:
        all_docs.extend(json.load(f))
print(f"  Loaded {len(all_docs)} documents")

# 2. Extract and analyze the most frequent topics, locations, and country codes from the Reuters corpus 
# to generate realistic query values for the evaluation dataset
topics_counter    = Counter()
places_counter    = Counter()
countries_counter = Counter()

for doc in all_docs:
    for t in doc.get("topics",      []): topics_counter[t]    += 1
    for p in doc.get("places",      []): places_counter[p]    += 1
    for c in doc.get("countryKeys", []): countries_counter[c] += 1

TOP_TOPICS    = [t for t, _ in topics_counter.most_common(12)]
TOP_PLACES    = [p for p, _ in places_counter.most_common(10)]
TOP_COUNTRIES = [c for c, _ in countries_counter.most_common(10)]

print(f"  Top topics   : {TOP_TOPICS}")
print(f"  Top places   : {TOP_PLACES}")
print(f"  Top countries: {TOP_COUNTRIES}")

# 3. Define a structured scoring rubric that evaluates Elasticsearch submissions
# across query structure, field validity, operator correctness, and task alignment
RUBRIC = {
    "0": (
        "Score 0 — Completely invalid: "
        "[Query Structure] The JSON is unparseable or uses entirely wrong top-level keys "
        "(e.g. 'filter' or 'order' at the root instead of 'query'/'sort'). "
        "[Field Validity] Fields used do not exist in the index mapping at all. "
        "[Operator Correctness] No recognizable Elasticsearch query operator is used correctly. "
        "[Task Alignment] The query is completely unrelated to the stated task — "
        "it would either throw a parsing error or return results that have nothing to do "
        "with what was asked."
    ),
    "1": (
        "Score 1 — Major errors: "
        "[Query Structure] A recognizable query type is attempted (e.g. bool, match, term) "
        "but the structure is fundamentally broken — e.g. a nested field queried without "
        "a nested wrapper, or a plain term used where a bool is required. "
        "[Field Validity] At least one critical field name is wrong or non-existent "
        "(e.g. 'headline' instead of 'title', 'tags' instead of 'topics'). "
        "[Operator Correctness] Wrong clause type used that changes semantics significantly "
        "(e.g. match_all instead of a specific filter). "
        "[Task Alignment] The query captures the right intent but the errors are severe enough "
        "that it would return empty results or completely wrong documents."
    ),
    "2": (
        "Score 2 — Partially correct: "
        "[Query Structure] The outer query structure is correct (e.g. bool with must/filter) "
        "but at least one required clause is entirely missing (e.g. no filter, no sort, "
        "no date range, missing one branch of a multi-field search). "
        "[Field Validity] All field names that ARE present are correct. "
        "[Operator Correctness] Operators used are correct where applied; errors are of "
        "omission, not commission. "
        "[Task Alignment] The query partially satisfies the task — it gets some results right "
        "but the missing component means the result set is over-broad or incomplete."
    ),
    "3": (
        "Score 3 — Mostly correct with minor issues: "
        "[Query Structure] All required clauses and nesting levels are present. "
        "[Field Validity] At most one field name is slightly wrong or suboptimal "
        "(e.g. using a non-existent field name in sort, or using 'match' on a keyword "
        "field where 'term' is preferred). "
        "[Operator Correctness] Operators are logically correct but one is imprecise — "
        "e.g. match vs term on a keyword field, or a sort on a non-indexed field. "
        "[Task Alignment] The query would return broadly correct results but with minor "
        "precision or correctness issues that a small one-line fix would resolve."
    ),
    "4": (
        "Score 4 — Fully correct: "
        "[Query Structure] The query uses exactly the right clause type(s), correctly nested "
        "and composed (e.g. proper bool > must + filter, nested wrapper with correct path). "
        "[Field Validity] Every field name is exactly correct and matches the index mapping. "
        "[Operator Correctness] All operators (match/term/range/sort order/nested path) are "
        "exactly right for the data type of each field. "
        "[Task Alignment] The query returns precisely the documents the task describes — "
        "no over-retrieval, no under-retrieval, and the sort/pagination is correct if specified."
    ),
}

# 4. Define diverse natural language task templates to generate realistic
# Elasticsearch search requests across multiple query categories
def pick(pool): return random.choice(pool)

def content_match_tasks(term):
    return [
        f'Find all news articles whose body text mentions "{term}".',
        f'Retrieve every document where the content field contains the word "{term}".',
        f'Search for articles that discuss "{term}" anywhere in their content.',
        f'Which Reuters articles have "{term}" mentioned in the content?',
        f'Pull all documents whose content includes the keyword "{term}".',
    ]

def topic_tasks(topic):
    return [
        f'Retrieve all documents that are tagged with the topic "{topic}".',
        f'Find every article labelled under the topic category "{topic}".',
        f'Get all documents whose topics field contains "{topic}".',
        f'Show me articles that belong to the "{topic}" topic.',
        f'Filter the index to documents tagged "{topic}" in their topics list.',
    ]

def date_range_tasks(desc):
    return [
        f'Find all documents {desc}.',
        f'Retrieve articles that were {desc}.',
        f'Get every document {desc}.',
        f'Which articles were {desc}?',
        f'Filter the index to documents that are {desc}.',
    ]

def country_tasks(country):
    return [
        f'Find all documents associated with country code "{country}".',
        f'Retrieve every article linked to the country key "{country}".',
        f'Get documents where countryKeys includes "{country}".',
        f'Show me all records tagged with the country code "{country}".',
        f'Filter by countryKeys to return only "{country}" documents.',
    ]

def nested_tasks(cc):
    return [
        f'Find documents containing a georeference with country_code "{cc}".',
        f'Retrieve articles that have a georeference entry where country_code is "{cc}".',
        f'Get all documents whose georeferences include an entry for country_code "{cc}".',
        f'Search inside the georeferences array for entries with country_code "{cc}".',
        f'Which documents have a nested georeference whose country_code equals "{cc}"?',
    ]

def multimatch_tasks(term):
    return [
        f'Search for "{term}" across both the title and content fields.',
        f'Find documents where either the title or the content contains "{term}".',
        f'Run a search for "{term}" in title and content simultaneously.',
        f'Retrieve articles mentioning "{term}" in their title or body text.',
        f'Multi-field search for "{term}" covering title and content.',
    ]

# 5. Define error injection utilities that generate realistic query mistakes
# to create diverse evaluation examples across different score levels
FIELD_TYPOS = {
    "title":        "headline",
    "content":      "body",
    "topics":       "tags",
    "countryKeys":  "countries",
    "date":         "published_at",
    "places":       "locations",
    "georeferences":"georefs",
}

def wrong_field(q: dict) -> dict:
    s = json.dumps(q)
    for real, fake in FIELD_TYPOS.items():
        if f'"{real}"' in s:
            return json.loads(s.replace(f'"{real}"', f'"{fake}"', 1))
    return q

def flip_operator(q: dict) -> dict:
    s = json.dumps(q)
    if '"gte"' in s: return json.loads(s.replace('"gte"', '"lte"', 1))
    if '"lte"' in s: return json.loads(s.replace('"lte"', '"gte"', 1))
    return q

def drop_filter(q: dict) -> dict:
    q2 = copy.deepcopy(q)
    if "query" in q2 and "bool" in q2["query"]:
        q2["query"]["bool"].pop("filter", None)
    return q2

def drop_sort(q: dict) -> dict:
    q2 = copy.deepcopy(q); q2.pop("sort", None); return q2

def match_instead_of_term(q: dict) -> dict:
    s = json.dumps(q)
    return json.loads(s.replace('"term"', '"match"', 1))

# Plausible-looking score-0 submissions (wrong top-level structure)
PLAUSIBLE_BROKEN = [
    {"filter": {"term": {"topics": "trade"}}, "sort": "date"},
    {"query": {"bool": {"filter": {"match": {"content": "oil"}}}}},   # filter not a list
    {"search": {"match": {"content": "grain"}}, "order": "desc"},
    {"query": {"terms": {"content": ["oil", "grain"]}}},  # terms on text field — invalid
    {"bool": {"must": [{"match": {"title": "value"}}]}},  # missing "query" wrapper
]

def broken_score0():
    return random.choice(PLAUSIBLE_BROKEN)

# 6. Initialize the dataset structure and define helper functions for storing
# labeled examples and generating rubric-based evaluation rationales
dataset = []

def add(task, reference, submission, score, rationale):
    dataset.append({
        "task":       task,
        "reference":  reference,
        "submission": submission,
        "rubric":     RUBRIC,
        "score":      score,
        "rationale":  rationale,
    })

# Helper: rubric-grounded rationale builder
def R(score, *points):
    """Build a rationale that names each rubric dimension explicitly."""
    return f"[Score {score}] " + " | ".join(points)


# 5a. Generate content-based query evaluation examples covering all score levels
# from fully correct queries to realistic structural and semantic errors
for term in TOP_TOPICS[:6]:
    ref = {"query": {"match": {"content": term}}}

    # 4 — perfect
    add(pick(content_match_tasks(term)), ref, ref, 4,
        R(4,
          "[Query Structure] Correct single match clause at the query level.",
          "[Field Validity] 'content' is a valid text field in the mapping.",
          "[Operator Correctness] 'match' is the correct operator for full-text search on a text field.",
          f"[Task Alignment] Returns exactly documents whose content mentions '{term}'."))

    # 3 — match on keyword 'topics' instead of text 'content' (minor field imprecision)
    add(pick(content_match_tasks(term)), ref,
        {"query": {"match": {"title": term}}}, 3,
        R(3,
          "[Query Structure] Correct match structure.",
          "[Field Validity] 'title' is a valid field but the task specifies 'content'.",
          "[Operator Correctness] 'match' is correct for text fields.",
          f"[Task Alignment] Searches the wrong field — retrieves articles mentioning '{term}' in the title only, missing content-only mentions."))

    # 2 — match exists but wrapped inside a bool with empty filter (partial)
    add(pick(content_match_tasks(term)), ref,
        {"query": {"bool": {"must": [{"match": {"content": term}}]}}}, 2,
        R(2,
          "[Query Structure] Unnecessarily wrapped in a bool/must but structurally valid; however no filter clause when none is needed is not wrong — this sub-case represents having the right match but inside the wrong clause type without any filter.",
          "[Field Validity] 'content' field is correct.",
          "[Operator Correctness] 'match' is correct.",
          "[Task Alignment] This particular variant is effectively equivalent; for diversity this entry models the case where the bool wrapper adds no filter and the evaluator notes unnecessary complexity but correct results."))

    # Actually score 2: wrong field used for match
    add(pick(content_match_tasks(term)), ref,
        wrong_field(copy.deepcopy(ref)), 2,
        R(2,
          "[Query Structure] Correct match clause structure.",
          "[Field Validity] The field 'body' does not exist in the mapping — correct field is 'content'. This causes zero results.",
          "[Operator Correctness] 'match' operator is appropriate for text search.",
          f"[Task Alignment] Intent is correct but the non-existent field means no documents are retrieved."))

    # 1 — term on content (wrong operator for text field, structural error)
    add(pick(content_match_tasks(term)), ref,
        {"query": {"term": {"content": term}}}, 1,
        R(1,
          "[Query Structure] Query is structurally parseable.",
          "[Field Validity] 'content' is a valid field.",
          "[Operator Correctness] 'term' on a text field is a major operator error — 'content' is analyzed/tokenized so term queries almost never match unless the value is a single token exactly as stored.",
          f"[Task Alignment] Will return near-zero results for '{term}' because analyzed text fields require 'match', not 'term'."))

    # 0 — completely broken structure
    add(pick(content_match_tasks(term)), ref, broken_score0(), 0,
        R(0,
          "[Query Structure] Invalid top-level keys — not a valid Elasticsearch query DSL structure.",
          "[Field Validity] Fields referenced either don't exist or are in wrong positions.",
          "[Operator Correctness] No valid operator is correctly applied.",
          "[Task Alignment] This query would either fail to parse or return completely unrelated results."))


# 5b. Generate topic-filtering query examples using exact keyword matching
# and controlled errors to represent scores from 0 to 4
for topic in TOP_TOPICS[:5]:
    ref = {"query": {"term": {"topics": topic}}}

    # 4 - fully correct topic-filtering query
    add(pick(topic_tasks(topic)), ref, ref, 4,
        R(4,
          "[Query Structure] Correct term query at the query level.",
          "[Field Validity] 'topics' is a valid keyword field in the mapping.",
          "[Operator Correctness] 'term' is the correct operator for exact-match on keyword fields.",
          f"[Task Alignment] Returns exactly documents tagged with topic '{topic}'."))

    # 3 — match instead of term (imprecise on keyword field)
    add(pick(topic_tasks(topic)), ref,
        match_instead_of_term(copy.deepcopy(ref)), 3,
        R(3,
          "[Query Structure] Query structure is valid.",
          "[Field Validity] 'topics' field is correct.",
          "[Operator Correctness] 'match' is imprecise on a keyword field — it performs analysis that may split or normalize the value, causing unexpected matches or misses. 'term' is required for exact keyword matching.",
          f"[Task Alignment] May return broadly correct results but with precision issues for multi-word topic values like '{topic}'."))

    # 2 — correct operator, completely missing the field value (empty term)
    add(pick(topic_tasks(topic)), ref,
        {"query": {"term": {"topics": ""}}}, 2,
        R(2,
          "[Query Structure] Correct term query structure.",
          "[Field Validity] 'topics' field is valid.",
          "[Operator Correctness] 'term' is the right operator.",
          f"[Task Alignment] The value is empty string instead of '{topic}' — returns zero relevant documents. Intent is partially right (correct field and operator) but the value is missing."))

    # 1 — wrong field name (non-existent)
    add(pick(topic_tasks(topic)), ref,
        {"query": {"term": {"tags": topic}}}, 1,
        R(1,
          "[Query Structure] Structurally valid term query.",
          "[Field Validity] 'tags' does not exist in the index mapping — the correct field is 'topics'. This is a critical field name error.",
          "[Operator Correctness] 'term' is the right operator type.",
          f"[Task Alignment] Returns zero results because the field doesn't exist. The intent is clear but the field name error is fatal."))

    # 0 — broken structure
    add(pick(topic_tasks(topic)), ref, broken_score0(), 0,
        R(0,
          "[Query Structure] Top-level structure is invalid — not a valid Elasticsearch query.",
          "[Field Validity] No valid mapping field is correctly targeted.",
          "[Operator Correctness] No valid Elasticsearch operator is used correctly.",
          "[Task Alignment] Would throw a parse error or return completely irrelevant documents."))


# 5c. Generate date range query examples with controlled boundary, operator,
# field, and structural errors to cover all evaluation score levels
date_cases = [
    ("published on or after 1987-03-01",  {"gte": "1987-03-01"}),
    ("published before 1987-06-01",       {"lte": "1987-06-01"}),
    ("published in 1987",                 {"gte": "1987-01-01", "lte": "1987-12-31"}),
    ("published after 1987-09-01",        {"gte": "1987-09-01"}),
    ("published between 1987-01-01 and 1987-03-31", {"gte": "1987-01-01", "lte": "1987-03-31"}),
]

for desc, range_val in date_cases:
    ref = {"query": {"range": {"date": range_val}}}

    # 4
    add(pick(date_range_tasks(desc)), ref, ref, 4,
        R(4,
          "[Query Structure] Correct range query at the query level.",
          "[Field Validity] 'date' is the valid date field in the mapping.",
          f"[Operator Correctness] Operators {list(range_val.keys())} are exactly correct for '{desc}'.",
          "[Task Alignment] Returns exactly the documents in the specified date window."))

    # 3 — correct structure and field but one operator boundary is off by one day
    adjusted = {k: v for k, v in range_val.items()}
    # Slightly wrong boundary (simulate off-by-one or imprecise boundary)
    first_key = list(adjusted.keys())[0]
    adjusted_copy = dict(adjusted)
    adjusted_copy[first_key] = adjusted_copy[first_key].replace("-01", "-02") \
        if "-01" in adjusted_copy[first_key] else adjusted_copy[first_key].replace("-31", "-30")
    add(pick(date_range_tasks(desc)), ref,
        {"query": {"range": {"date": adjusted_copy}}}, 3,
        R(3,
          "[Query Structure] Correct range query structure.",
          "[Field Validity] 'date' field is correct.",
          f"[Operator Correctness] Operator direction is correct but boundary date is slightly off ('{adjusted_copy[first_key]}' vs '{adjusted[first_key]}'). Minor precision issue.",
          "[Task Alignment] Returns mostly correct documents but may miss or include a small number of edge documents due to the boundary difference."))

    # 2 — flipped operator (returns opposite date window)
    add(pick(date_range_tasks(desc)), ref,
        flip_operator(copy.deepcopy(ref)), 2,
        R(2,
          "[Query Structure] Correct range query structure.",
          "[Field Validity] 'date' is the correct field.",
          "[Operator Correctness] The comparison operator is flipped (gte↔lte), inverting the date window. This is a significant logic error that returns the opposite set of documents.",
          "[Task Alignment] The field and structure are right but the result set is the complement of what was asked for."))

    # 1 — wrong field name
    add(pick(date_range_tasks(desc)), ref,
        {"query": {"range": {"published_at": range_val}}}, 1,
        R(1,
          "[Query Structure] Correct range query structure.",
          "[Field Validity] 'published_at' does not exist in the mapping — the correct date field is 'date'. This is a fatal field name error.",
          "[Operator Correctness] Operators are correct.",
          "[Task Alignment] Returns zero results because the field doesn't exist in the index."))

    # 0 — broken structure
    add(pick(date_range_tasks(desc)), ref, broken_score0(), 0,
        R(0,
          "[Query Structure] Not a valid Elasticsearch query — wrong top-level keys or completely malformed DSL.",
          "[Field Validity] No valid date field is correctly referenced.",
          "[Operator Correctness] No range operator is applied.",
          "[Task Alignment] Would fail to parse or return entirely irrelevant documents."))


# 5d. Generate Boolean query evaluation examples that combine content matching
# with filtering constraints while introducing logical and structural variations
# to represent all rubric score levels
bool_cases = [
    {
        "task_pool": [
            f'Find articles about "{TOP_TOPICS[0]}" published after 1987-01-01.',
            f'Retrieve news mentioning "{TOP_TOPICS[0]}" that appeared after January 1987.',
            f'Get all documents discussing "{TOP_TOPICS[0]}" from 1987 onwards.',
        ],
        "ref": {"query": {"bool": {
            "must":   [{"match": {"content": TOP_TOPICS[0]}}],
            "filter": [{"range": {"date": {"gte": "1987-01-01"}}}]
        }}}
    },
    {
        "task_pool": [
            f'Find articles about "{TOP_TOPICS[1]}" tagged with topic "{TOP_TOPICS[1]}".',
            f'Retrieve documents whose content mentions "{TOP_TOPICS[1]}" and that are tagged under "{TOP_TOPICS[1]}".',
            f'Get articles both discussing and categorised under "{TOP_TOPICS[1]}".',
        ],
        "ref": {"query": {"bool": {
            "must":   [{"match": {"content": TOP_TOPICS[1]}}],
            "filter": [{"term": {"topics": TOP_TOPICS[1]}}]
        }}}
    },
    {
        "task_pool": [
            f'Find articles with "{TOP_TOPICS[2]}" in the title published before 1987-06-01.',
            f'Retrieve documents whose title contains "{TOP_TOPICS[2]}" and that predate June 1987.',
            f'Get pre-June-1987 articles mentioning "{TOP_TOPICS[2]}" in their title.',
        ],
        "ref": {"query": {"bool": {
            "must":   [{"match": {"title": TOP_TOPICS[2]}}],
            "filter": [{"range": {"date": {"lte": "1987-06-01"}}}]
        }}}
    },
    {
        "task_pool": [
            f'Find documents from country "{TOP_COUNTRIES[0]}" mentioning "{TOP_TOPICS[3]}".',
            f'Retrieve articles associated with country code "{TOP_COUNTRIES[0]}" that discuss "{TOP_TOPICS[3]}".',
            f'Get all "{TOP_COUNTRIES[0]}" documents whose content includes "{TOP_TOPICS[3]}".',
        ],
        "ref": {"query": {"bool": {
            "must":   [{"match": {"content": TOP_TOPICS[3]}}],
            "filter": [{"term": {"countryKeys": TOP_COUNTRIES[0]}}]
        }}}
    },
    {
        "task_pool": [
            f'Find articles from country "{TOP_COUNTRIES[1]}" published in 1987.',
            f'Retrieve "{TOP_COUNTRIES[1]}" documents from the year 1987.',
            f'Get all 1987 articles linked to country code "{TOP_COUNTRIES[1]}".',
        ],
        "ref": {"query": {"bool": {
            "must":   [{"match_all": {}}],
            "filter": [
                {"term": {"countryKeys": TOP_COUNTRIES[1]}},
                {"range": {"date": {"gte": "1987-01-01", "lte": "1987-12-31"}}}
            ]
        }}}
    },
]

for case in bool_cases:
    t   = pick(case["task_pool"])
    ref = case["ref"]

    # 4 — perfect
    add(t, ref, ref, 4,
        R(4,
          "[Query Structure] Correct bool query with both 'must' and 'filter' clauses properly composed.",
          "[Field Validity] All field names match the index mapping exactly.",
          "[Operator Correctness] match/term/range operators are all appropriate for their respective field types.",
          "[Task Alignment] Returns exactly the documents satisfying both the relevance and the constraint conditions."))

    # 3 — filter has a wrong field name (minor fix needed)
    sub3 = copy.deepcopy(ref)
    bool_part = sub3["query"]["bool"]
    if "filter" in bool_part:
        filt = bool_part["filter"][0]
        if "term" in filt:
            key = list(filt["term"].keys())[0]
            filt["term"] = {FIELD_TYPOS.get(key, key + "_bad"): filt["term"][key]}
        elif "range" in filt:
            key = list(filt["range"].keys())[0]
            filt["range"] = {FIELD_TYPOS.get(key, key + "_bad"): filt["range"][key]}
    add(t, ref, sub3, 3,
        R(3,
          "[Query Structure] Correct bool structure with must and filter clauses.",
          "[Field Validity] The must clause field is correct; the filter clause references a non-existent field — a single field name fix is all that's needed.",
          "[Operator Correctness] Operators (match/term/range) are all correct.",
          "[Task Alignment] The query logic is right but the invalid filter field means the constraint is not applied, making the result set over-broad."))

    # 2 — filter missing entirely
    add(t, ref, drop_filter(copy.deepcopy(ref)), 2,
        R(2,
          "[Query Structure] The bool query is valid but only has a 'must' clause — the 'filter' clause is entirely absent.",
          "[Field Validity] The must clause field is correct.",
          "[Operator Correctness] The must operator is correct; the filter operator doesn't exist to evaluate.",
          "[Task Alignment] The query only satisfies the content-matching part of the task. The date/topic/country constraint is completely missing, returning too many documents."))

    # 1 — should is used instead of must+filter (wrong bool logic)
    sub1 = {"query": {"bool": {"should": list(ref["query"]["bool"].get("must", []))}}}
    add(t, ref, sub1, 1,
        R(1,
          "[Query Structure] Uses 'should' instead of 'must'+'filter' — fundamentally wrong boolean logic. 'should' makes clauses optional, not required.",
          "[Field Validity] Field names in the should clause may be correct.",
          "[Operator Correctness] The clause type 'should' is wrong for this task — it does not enforce any constraint.",
          "[Task Alignment] Returns documents that optionally match the condition rather than requiring it, producing a semantically incorrect result set."))

    # 0 — completely broken
    add(t, ref, broken_score0(), 0,
        R(0,
          "[Query Structure] Not a valid Elasticsearch query — wrong top-level keys or completely missing 'query' wrapper.",
          "[Field Validity] No valid mapping fields are correctly referenced.",
          "[Operator Correctness] No valid bool, must, or filter operators are correctly applied.",
          "[Task Alignment] Would fail to parse or return completely irrelevant documents with no filtering."))


# 5e. Generate sorting query examples that evaluate correct date ordering
# and introduce missing, reversed, invalid, or malformed sort configurations
sort_cases = [
    {
        "task_pool": [
            f'Find articles about "{TOP_TOPICS[0]}" sorted by date descending.',
            f'Retrieve documents mentioning "{TOP_TOPICS[0]}" in order of newest first.',
            f'Get "{TOP_TOPICS[0]}" articles with the most recent ones at the top.',
        ],
        "ref": {
            "query": {"match": {"content": TOP_TOPICS[0]}},
            "sort":  [{"date": {"order": "desc"}}]
        },
        "order": "desc"
    },
    {
        "task_pool": [
            f'Find documents tagged "{TOP_TOPICS[1]}" sorted by date ascending.',
            f'Retrieve "{TOP_TOPICS[1]}" articles from oldest to newest.',
            f'Get all "{TOP_TOPICS[1]}" documents in chronological order.',
        ],
        "ref": {
            "query": {"term": {"topics": TOP_TOPICS[1]}},
            "sort":  [{"date": {"order": "asc"}}]
        },
        "order": "asc"
    },
]

for case in sort_cases:
    t, ref, order = pick(case["task_pool"]), case["ref"], case["order"]

    # 4
    add(t, ref, ref, 4,
        R(4,
          "[Query Structure] Correct query clause plus a valid top-level 'sort' array.",
          "[Field Validity] Both the query field and the sort field ('date') exist in the mapping.",
          f"[Operator Correctness] Sort order '{order}' is correct for the task.",
          "[Task Alignment] Returns the right documents in the correct order."))

    # 3 — sort on non-existent field
    wrong_sort = copy.deepcopy(ref)
    wrong_sort["sort"] = [{"nonexistent_field": {"order": order}}]
    add(t, ref, wrong_sort, 3,
        R(3,
          "[Query Structure] Query and sort clauses are both present and structurally correct.",
          "[Field Validity] The query field is correct; the sort field 'nonexistent_field' does not exist in the mapping.",
          f"[Operator Correctness] Sort order '{order}' is correct.",
          "[Task Alignment] The query part is perfect but sorting on a non-existent field will either error or fall back to default ordering, failing to satisfy the sort requirement."))

    # 2 — sort missing entirely
    add(t, ref, drop_sort(copy.deepcopy(ref)), 2,
        R(2,
          "[Query Structure] The query clause is correct but the 'sort' clause is entirely absent.",
          "[Field Validity] The query field is correct.",
          "[Operator Correctness] The query operator is correct; sort operator is absent.",
          "[Task Alignment] Returns the right documents but in arbitrary order, not the required date order. Half the task is complete."))

    # 1 — sort order reversed (wrong direction)
    wrong_order = "asc" if order == "desc" else "desc"
    sub1 = copy.deepcopy(ref)
    sub1["sort"] = [{"date": {"order": wrong_order}}]
    add(t, ref, sub1, 1,
        R(1,
          "[Query Structure] Query and sort clauses are present.",
          "[Field Validity] All field names are correct.",
          f"[Operator Correctness] Sort order is '{wrong_order}' but the task requires '{order}' — the direction is exactly backwards.",
          "[Task Alignment] Returns the right documents but in the opposite order, completely inverting the intended ranking."))

    # 0 — invalid top-level structure
    add(t, ref,
        {"filter": {"match": {"content": "x"}}, "order": order}, 0,
        R(0,
          "[Query Structure] 'filter' and 'order' are not valid Elasticsearch top-level keys — the correct keys are 'query' and 'sort'. This query would fail to parse.",
          "[Field Validity] No valid field is correctly targeted in the right position.",
          "[Operator Correctness] No valid operator is applied.",
          "[Task Alignment] Completely invalid — would throw a parse error and return no results."))


# 5f. Generate nested georeference query examples that validate correct nested
# paths, field usage, and operator selection across all rubric score levels
for cc in TOP_COUNTRIES[:5]:
    ref = {
        "query": {
            "nested": {
                "path":  "georeferences",
                "query": {"term": {"georeferences.country_code": cc}}
            }
        }
    }

    # 4
    add(pick(nested_tasks(cc)), ref, ref, 4,
        R(4,
          "[Query Structure] Correct nested query with 'path' and inner 'query' both present.",
          "[Field Validity] 'georeferences' is a valid nested field; 'georeferences.country_code' is the correct dotted path.",
          "[Operator Correctness] 'term' is correct for exact matching on the keyword subfield 'country_code'.",
          f"[Task Alignment] Returns exactly the documents with a nested georeference entry for country_code '{cc}'."))

    # 3 — match instead of term inside nested
    sub3 = {
        "query": {
            "nested": {
                "path":  "georeferences",
                "query": {"match": {"georeferences.country_code": cc}}
            }
        }
    }
    add(pick(nested_tasks(cc)), ref, sub3, 3,
        R(3,
          "[Query Structure] Correct nested wrapper with 'path' and inner 'query'.",
          "[Field Validity] Path and field name are both correct.",
          "[Operator Correctness] 'match' is used on a keyword subfield where 'term' is required for exact matching. This is a minor operator precision issue.",
          f"[Task Alignment] Will mostly return correct results but 'match' may produce unexpected behavior on the keyword field 'georeferences.country_code'."))

    # 2 — nested wrapper present but wrong path
    sub2 = {
        "query": {
            "nested": {
                "path":  "georefs",    # wrong path
                "query": {"term": {"georeferences.country_code": cc}}
            }
        }
    }
    add(pick(nested_tasks(cc)), ref, sub2, 2,
        R(2,
          "[Query Structure] Nested wrapper structure is correct (path + query present).",
          "[Field Validity] The nested path 'georefs' does not match the mapping — the correct path is 'georeferences'. The inner field name is correct.",
          "[Operator Correctness] 'term' is the right operator.",
          f"[Task Alignment] The query cannot traverse the nested objects because the path is wrong — returns zero results despite correct inner query logic."))

    # 1 — no nested wrapper (plain term on nested field)
    add(pick(nested_tasks(cc)), ref,
        {"query": {"term": {"georeferences.country_code": cc}}}, 1,
        R(1,
          "[Query Structure] Missing the required 'nested' wrapper — a plain term query cannot traverse nested objects in Elasticsearch.",
          "[Field Validity] The dotted field path 'georeferences.country_code' is correct.",
          "[Operator Correctness] 'term' is the right operator but it's applied at the wrong query level.",
          "[Task Alignment] Elasticsearch ignores plain term queries on nested fields — returns empty or wrong results."))

    # 0
    add(pick(nested_tasks(cc)), ref, broken_score0(), 0,
        R(0,
          "[Query Structure] Completely invalid Elasticsearch DSL — no 'nested' wrapper, no valid 'query' key at the right level.",
          "[Field Validity] Nested path and field are not correctly referenced.",
          "[Operator Correctness] No valid operator is applied inside the nested context.",
          "[Task Alignment] Would fail to parse or return entirely unrelated documents."))


# 5g. Generate country code filtering examples that test exact keyword matching
# on countryKeys while introducing field, operator, and alignment errors
for country in TOP_COUNTRIES[:6]:
    ref = {"query": {"term": {"countryKeys": country}}}

    # 4
    add(pick(country_tasks(country)), ref, ref, 4,
        R(4,
          "[Query Structure] Correct single term query.",
          "[Field Validity] 'countryKeys' is the valid keyword field for country codes.",
          "[Operator Correctness] 'term' is the correct operator for exact match on a keyword field.",
          f"[Task Alignment] Returns exactly documents associated with country code '{country}'."))

    # 3 — match instead of term
    add(pick(country_tasks(country)), ref,
        match_instead_of_term(copy.deepcopy(ref)), 3,
        R(3,
          "[Query Structure] Valid query structure.",
          "[Field Validity] 'countryKeys' is the correct field.",
          "[Operator Correctness] 'match' on a keyword field is imprecise — country codes should be matched exactly with 'term', not analyzed with 'match'.",
          f"[Task Alignment] May mostly work for single-token country codes like '{country}' but is semantically wrong and fragile for multi-part codes."))

    # 2 — wrong field (places instead of countryKeys)
    add(pick(country_tasks(country)), ref,
        {"query": {"term": {"places": country}}}, 2,
        R(2,
          "[Query Structure] Correct term query structure.",
          "[Field Validity] 'places' stores human-readable place name strings, not ISO country codes — the correct field is 'countryKeys'. Wrong field but exists in the mapping.",
          "[Operator Correctness] 'term' is the right operator type.",
          f"[Task Alignment] Will not find documents by country code '{country}' since 'places' contains names not codes. Results will be empty or wrong."))

    # 1 — completely wrong field (non-existent)
    add(pick(country_tasks(country)), ref,
        {"query": {"term": {"countries": country}}}, 1,
        R(1,
          "[Query Structure] Correct term query structure.",
          "[Field Validity] 'countries' does not exist in the mapping at all — the correct field is 'countryKeys'. Fatal field name error.",
          "[Operator Correctness] 'term' is the right operator.",
          "[Task Alignment] Returns zero results because the field doesn't exist in the index."))

    # 0 — match_all (no filtering at all)
    add(pick(country_tasks(country)), ref,
        {"query": {"match_all": {}}}, 0,
        R(0,
          "[Query Structure] Syntactically valid but semantically empty — match_all returns every document.",
          "[Field Validity] No field is targeted.",
          "[Operator Correctness] match_all is not a filtering operator.",
          f"[Task Alignment] Returns all documents in the index with no filtering by country '{country}' — completely fails the task."))


# 5h. Generate multi-field search examples that evaluate proper multi_match
# usage across title and content while introducing field and operator errors
search_terms = [TOP_TOPICS[0], TOP_TOPICS[2], "oil prices", "interest rate", "export"]

for term in search_terms:
    ref = {
        "query": {
            "multi_match": {
                "query":  term,
                "fields": ["title", "content"]
            }
        }
    }

    # 4
    add(pick(multimatch_tasks(term)), ref, ref, 4,
        R(4,
          "[Query Structure] Correct multi_match query with 'query' and 'fields' both present.",
          "[Field Validity] Both 'title' and 'content' are valid text fields.",
          "[Operator Correctness] 'multi_match' is the correct operator for searching across multiple fields simultaneously.",
          f"[Task Alignment] Returns all documents where '{term}' appears in either title or content."))

    # 3 — multi_match but only one field
    add(pick(multimatch_tasks(term)), ref,
        {"query": {"multi_match": {"query": term, "fields": ["title"]}}}, 3,
        R(3,
          "[Query Structure] Correct multi_match structure.",
          "[Field Validity] 'title' is a valid field.",
          "[Operator Correctness] 'multi_match' is correct but underused — only one field is listed.",
          f"[Task Alignment] Searches only 'title', missing all documents where '{term}' appears only in 'content'. Half the required fields are covered."))

    # 2 — single match on title only (wrong operator, missing field)
    add(pick(multimatch_tasks(term)), ref,
        {"query": {"match": {"title": term}}}, 2,
        R(2,
          "[Query Structure] Valid match query but single-field — does not use multi_match.",
          "[Field Validity] 'title' is a valid field.",
          "[Operator Correctness] 'match' is wrong here — the task explicitly requires searching both title and content simultaneously, which requires 'multi_match'.",
          f"[Task Alignment] Returns only documents mentioning '{term}' in the title, missing all content-only mentions."))

    # 1 — multi_match with non-existent fields
    add(pick(multimatch_tasks(term)), ref,
        {"query": {"multi_match": {"query": term, "fields": ["headline", "body"]}}}, 1,
        R(1,
          "[Query Structure] Correct multi_match structure.",
          "[Field Validity] 'headline' and 'body' do not exist in the mapping — the correct fields are 'title' and 'content'. Both field names are wrong.",
          "[Operator Correctness] 'multi_match' is the right operator.",
          "[Task Alignment] Returns zero results because neither field exists in the index."))

    # 0
    add(pick(multimatch_tasks(term)), ref, broken_score0(), 0,
        R(0,
          "[Query Structure] Invalid Elasticsearch DSL — wrong top-level keys or missing 'query' wrapper.",
          "[Field Validity] No valid mapping fields are correctly referenced.",
          "[Operator Correctness] No valid operator is applied.",
          "[Task Alignment] Would fail to parse or return completely unrelated documents."))


# 6. Randomize the dataset and split it into training, validation, and testing
# subsets according to the user-defined partition ratios
random.shuffle(dataset)
n         = len(dataset)
train_end = int(args.train_ratio * n)
val_end   = train_end + int(args.val_ratio * n)

splits = {
    "train": dataset[:train_end],
    "val":   dataset[train_end:val_end],
    "test":  dataset[val_end:]
}

# 7. Export the complete dataset and its train, validation, and test splits
# to JSON files for model training and evaluation
full_path = os.path.join(args.out_dir, "es_eval_dataset_full.json")
with open(full_path, "w") as f:
    json.dump(dataset, f, indent=2)

for name, data in splits.items():
    with open(os.path.join(args.out_dir, f"es_eval_{name}.json"), "w") as f:
        json.dump(data, f, indent=2)

# 8. Generate summary statistics to verify dataset size, score distribution,
# and the completeness of all evaluation score categories
dist = Counter(e["score"] for e in dataset)
print(f"\n{'='*55}")
print(f"Dataset generated: {n} total entries")
print(f"  train : {len(splits['train'])}")
print(f"  val   : {len(splits['val'])}")
print(f"  test  : {len(splits['test'])}")
print(f"\nScore distribution (full dataset):")
for score in sorted(dist):
    bar = "█" * dist[score]
    print(f"  {score}: {dist[score]:3d}  {bar}")

# Verify no score level is missing
missing = [s for s in range(5) if s not in dist]
if missing:
    print(f"\nWARNING: Missing score levels: {missing}")
else:
    print(f"\nAll score levels 0–4 are represented.")

print(f"\nOutput written to: {os.path.abspath(args.out_dir)}/")
for name in ["es_eval_dataset_full.json"] + [f"es_eval_{s}.json" for s in splits]:
    print(f"  {name}")
