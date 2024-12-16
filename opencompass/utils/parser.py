import re

def extract_api(text, api_name="API", content_typ="json"):
    # Define regular expression pattern
    pattern = rf"<{api_name}>(.*?)</{api_name}>"
    # Match and extract with regular expressions
    matches = re.findall(pattern, text)
    extracted_content = []

    for match in matches:
        try:
            if content_typ == "json":
                json_obj = eval(match)[0]
            else:
                json_obj = match
            extracted_content.append(json_obj)
        except:
            extracted_content.append(None)

    return extracted_content


def extract_actions(text):
    json_str = extract_api(text)
    actions = [x["ActionName"] for x in json_str]
    actions = list(set(actions))
    return actions


def extract_actions_from_path(fpath):
    with open(fpath, "r") as f:
        text = f.read()
    actions = extract_actions(text)
    return actions

