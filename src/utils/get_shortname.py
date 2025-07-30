import json
import csv

with open("cancer_types.json", "r", encoding="utf-8") as f:
    cancer_types = json.load(f)

shortname_to_name = {
    entry["shortName"].upper(): entry["name"]
    for entry in cancer_types if entry.get("shortName")
}

mapped_data = []
for entry in cancer_types:
    name = entry.get("name", "")
    short_name = entry.get("shortName", "")
    parent = entry.get("parent", "")

    updated_parents = []
    if parent:
        for part in parent.split("/"):
            upper_part = part.upper()
            updated_parents.append(shortname_to_name.get(upper_part, part))
    updated_parent = " / ".join(updated_parents) if updated_parents else ""

    mapped_data.append({
        "name": name,
        "shortName": short_name,
        "parent": updated_parent
    })

with open("cancer_type_mapping.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["name", "shortName", "parent"])
    writer.writeheader()
    writer.writerows(mapped_data)

print("Mapping file saved as cancer_type_mapping.csv")
