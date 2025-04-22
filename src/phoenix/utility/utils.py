import json
import os


def write_to_config(path, **kwargs):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'config.json')

    with open(path, 'w') as file:
        json.dump(kwargs, file, indent=4)


def get_frag_name_from_id(frag_id):
    return str(frag_id)
