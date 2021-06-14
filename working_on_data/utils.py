import pandas as pd
import json
import re

def get_categories(entry):
    if "category" not in entry.keys(): return pd.Series()

    category_dict = json.loads(entry["category"])
    if "parent_name" in category_dict.keys():
        return pd.Series({
            "category": category_dict["parent_name"],
            "sub_category": category_dict["name"]
        })
    else:
        return pd.Series({
            "category": category_dict["name"]
        })


def get_urls(entry):
    return pd.Series({"project_url": json.loads(entry["urls"])["web"]["project"]})


def get_creator(entry):
    return pd.Series({"creator_id": int(re.search(r"(?<=\"id\":)\d+(?=,)", entry["creator"]).group(0))})
