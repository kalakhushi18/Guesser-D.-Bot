import requests
from bs4 import BeautifulSoup
import json
import re


straw_hat_pirates = [
    "Monkey_D._Luffy",
    "Roronoa_Zoro",
    "Nami",
    "Usopp",
    "Sanji",
    "Tony_Tony_Chopper",
    "Nico_Robin",
    "Franky",
    "Brook",
    "Jinbe"
]

url = "https://onepiece.fandom.com/wiki/Marshall_D._Teach"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

all_characters = {}
stats = {}


def extract_character_description(soup):
    content_div = soup.find("div", class_="mw-parser-output")
    description = ""

    if content_div:
        paragraphs = content_div.find_all("p", recursive=False)
        for para in paragraphs:
            # Skip empty or irrelevant lines
            text = para.get_text(strip=True)
            if text and len(text) > 80:  # you can tune this threshold
                description = text
                break

    return description


def clean_text(text):
    # Remove text inside brackets like [1], [12], etc.
    text = re.sub(r"\[\d+\]", ",", text)
    # Remove text in parentheses that contains Japanese characters
    text = re.sub(r"\([^()]*[\u3040-\u30ff\u4e00-\u9faf][^()]*\)", "", text)
    # Remove lingering ? symbols and extra spaces
    text = text.replace("?", "").replace("\u3000", " ").strip()
    # Normalize whitespace and semicolons
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;")

def parse_value_with_context(text):
    entries = []
    segments = re.split(r";|\n", text)
    for segment in segments:
        match = re.match(r"(?P<value>[\d.]+\s?\S*)\s*\(?(?P<context>[a-zA-Z\s-]+)?\)?", segment.strip())
        if match:
            entries.append({
                "value": match.group("value").strip(),
                "context": match.group("context").strip() if match.group("context") else ""
            })
    return entries

def parse_value_height(text):
    raw_lines = list(text.stripped_strings)
    entries = []

    for line in raw_lines:
        # Example: "172 cm (5'8\") (pre-timeskip)"
        match = re.match(r"(?P<value>\d{2,3}\s?cm\s?\([^()]+\))\s*\((?P<context>.+?)\)", line.strip())
        if match:
            value_clean = match.group("value").strip()
            context_clean = match.group("context").strip()

            entries.append({
                "value": value_clean,
                "context": context_clean
            })
    return entries

# Character name
name_tag = soup.find("h2", class_="pi-item pi-item-spacing pi-title pi-secondary-background")
character_name = name_tag.text.strip() if name_tag else "Unknown"

# Info box: section table
info_box = soup.find("section", class_="pi-item pi-group pi-border-color pi-collapse pi-collapse-open")

description = extract_character_description(soup)
stats["Description"] = clean_text(description)



label_exclude = ["japanese name", "romanized name"]
if info_box: 
    data_items = info_box.find_all("div", class_="pi-item pi-data pi-item-spacing pi-border-color")
    for item in data_items:
        label_tag = item.find("h3", class_="pi-data-label pi-secondary-font")
        value_tag = item.find("div", class_="pi-data-value pi-font")

        if label_tag and value_tag:

            for tag in value_tag(["sup", "ruby", "span", "i"]):
                tag.decompose()

            raw_text = ' '.join(value_tag.stripped_strings)
            label = clean_text(label_tag.get_text(strip=True).replace(":", ""))
            value = clean_text(raw_text)

            print("label:", label)

            if label.lower() in label_exclude:
                print("Excluded label:", label)
                continue

            elif label.lower() in ["age"]:
                stats[label] = parse_value_with_context(value)

            elif label.lower() in ["height"]:
                stats[label] = parse_value_height(value_tag)


            elif label.lower() == "bounty":
                bounty_entries = []
                # Extract all numbers like 3,000,000,000 or 500000000
                amounts = re.findall(r"[\d,]+", value)
                for i, amount in enumerate(amounts):
                    bounty_entries.append({
                        "value": amount.strip(),
                        "label": "current" if i == 0 else "previous"
                    })
                stats[label] = bounty_entries
            
            elif label.lower() == "alias":
                aliases = value.split()
                stats[label] = aliases

            else:
                stats[label] = value


all_characters[character_name] = stats


# Save to master JSON file
with open("marines_individual.json", "w", encoding="utf-8") as f:
    json.dump(all_characters, f, ensure_ascii=False, indent=2)


