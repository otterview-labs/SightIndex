import re

LABEL_CATALOG: dict[str, tuple[str, ...]] = {
    "category": ("person", "vehicle", "animal"),
    "object_type": ("pedestrian",),
    "gender": ("male", "female"),
    "age_group": ("child", "teenager", "young_adult", "middle_aged", "elderly"),
    "upper_color": (
        "white",
        "black",
        "grey",
        "blue",
        "red",
        "green",
        "yellow",
        "brown",
        "pink",
        "purple",
        "other",
    ),
    "lower_color": ("white", "black", "grey", "blue", "red", "green", "brown", "other"),
    "clothing_type": (
        "shorts",
        "skirt",
        "long_sleeve",
        "short_sleeve",
        "sleeveless",
        "jacket",
        "down_jacket",
        "hoodie",
        "suit",
        "shirt",
        "tshirt",
        "vest",
        "windbreaker",
        "sweater",
        "sportswear",
        "school_uniform",
        "long_skirt",
        "short_skirt",
        "pajamas",
        "police_uniform",
        "camouflage",
        "apron",
    ),
    "clothing_pattern": ("striped", "plaid", "printed", "solid"),
    "hair_style": (
        "straight",
        "curly",
        "ponytail",
        "bun",
        "braid",
        "bald",
        "short",
        "long",
        "dyed",
        "white",
    ),
    "hair_color": ("red", "blonde", "highlighted", "blue", "green"),
    "skin_tone": ("fair", "dark", "yellow"),
    "facial_hair": ("beard", "mustache"),
    "body_type": ("thin", "average", "heavy", "muscular", "tall", "short", "burly"),
    "accessory": (
        "mask",
        "glasses",
        "sunglasses",
        "hat",
        "helmet",
        "scarf",
        "tie",
        "earrings",
        "necklace",
        "bracelet",
        "ring",
    ),
    "carried_item": (
        "backpack",
        "shoulder_bag",
        "handbag",
        "luggage",
        "umbrella",
        "headphones",
        "phone",
        "walkie_talkie",
        "flashlight",
        "briefcase",
        "woven_bag",
    ),
    "footwear": (
        "black",
        "white",
        "brown",
        "sneakers",
        "leather_shoes",
        "boots",
        "sandals",
        "slippers",
    ),
    "action": (
        "walking",
        "running",
        "standing",
        "squatting",
        "cycling",
        "driving",
        "phone_calling",
        "smoking",
        "carrying_item",
    ),
    "posture": ("bending", "raising_hand", "leaning", "hands_in_pockets", "looking_back"),
    "ppe_safety_helmet": ("on", "off"),
    "ppe_safety_vest": ("on", "off"),
    "ppe_goggles": ("on", "off"),
    "ppe_gloves": ("on", "off"),
    "ppe_mask": ("on",),
    "ppe_harness": ("on", "off"),
    "seatbelt": (
        "on",
        "not_wearing",
        "driver_wearing",
        "driver_not_wearing",
        "passenger_wearing",
        "passenger_not_wearing",
    ),
    "vehicle_type": (
        "sedan",
        "suv",
        "mpv",
        "truck",
        "van",
        "bus",
        "minibus",
        "motorcycle",
        "electric_bike",
        "bicycle",
        "tricycle",
        "pickup",
        "special_vehicle",
        "offroad",
        "sports_car",
        "convertible",
        "hatchback",
    ),
    "special_vehicle": (
        "ambulance",
        "fire_truck",
        "police_car",
        "school_bus",
        "construction_vehicle",
        "dump_truck",
        "hazardous_material",
        "sanitation_truck",
        "postal_vehicle",
        "learner_vehicle",
    ),
    "vehicle_color": (
        "white",
        "black",
        "silver",
        "grey",
        "blue",
        "red",
        "green",
        "yellow",
        "brown",
        "orange",
        "purple",
        "gold",
        "multi_color",
    ),
    "vehicle_brand": (
        "volkswagen",
        "toyota",
        "honda",
        "nissan",
        "bmw",
        "mercedes",
        "audi",
        "byd",
        "changan",
        "geely",
        "haval",
        "wuling",
        "tesla",
        "buick",
        "hyundai",
        "kia",
    ),
    "plate_color": ("blue", "green", "yellow", "white", "black"),
    "vehicle_feature": (
        "sunroof",
        "roof_rack",
        "tinted_windows",
        "spare_tire",
        "decals",
        "modified",
        "taxi",
        "driving_school",
        "government",
        "new_energy",
        "temp_plate",
        "plate_covered",
        "no_plate",
    ),
    "vehicle_status": (
        "parked",
        "door_open",
        "trunk_open",
        "flat_tire",
        "windshield_cracked",
        "vehicle_damaged",
        "vehicle_smoke",
    ),
    "vehicle_light": (
        "headlights_on",
        "headlights_off",
        "brake_lights_on",
        "turn_signal_left",
        "turn_signal_right",
        "hazard_lights",
        "high_beam",
        "fog_light",
    ),
    "animal_type": (
        "dog",
        "cat",
        "bird",
        "cow",
        "horse",
        "sheep",
        "pig",
        "chicken",
        "duck",
        "rabbit",
        "squirrel",
        "fox",
        "deer",
        "wild_animal",
        "stray_animal",
        "other_animal",
    ),
    "animal_color": (
        "white",
        "black",
        "brown",
        "yellow",
        "grey",
        "spotted",
        "black_white",
        "yellow_white",
    ),
    "animal_size": ("small", "medium", "large", "giant"),
    "animal_behavior": (
        "walking",
        "running",
        "sitting",
        "lying",
        "eating",
        "leashed_animal",
        "animal_collar",
        "caged_animal",
        "injured_animal",
        "aggressive_animal",
        "animal_group",
    ),
    "animal_breed": (
        "golden_retriever",
        "labrador",
        "husky",
        "poodle",
        "corgi",
        "samoyed",
        "german_shepherd",
        "border_collie",
        "shiba_inu",
        "bulldog",
        "chinese_rural_dog",
        "tibetan_mastiff",
        "rottweiler",
        "doberman",
        "bully",
        "pitbull",
        "other_breed",
    ),
}

_FIELD_ALIASES = {
    "uppercolor": "upper_color",
    "upper-color": "upper_color",
    "lowercolor": "lower_color",
    "lower-color": "lower_color",
}

_VALUE_ALIASES = {
    "gray": "grey",
    "middle-age": "middle_aged",
    "middle age": "middle_aged",
}

_LABEL_PATTERN = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*[:.]\s*([A-Za-z0-9_ -]+)\s*$")
_EXCLUSIVE_FIELDS = {
    "gender",
    "age_group",
    "ppe_safety_helmet",
    "ppe_safety_vest",
    "ppe_goggles",
    "ppe_gloves",
    "ppe_mask",
    "ppe_harness",
    "seatbelt",
    "animal_size",
}


def normalize_label(raw_label: object) -> str | None:
    if not isinstance(raw_label, str):
        return None
    match = _LABEL_PATTERN.match(raw_label)
    if not match:
        return None
    field = _normalize_token(match.group(1))
    value = _normalize_token(match.group(2))
    field = _FIELD_ALIASES.get(field, field)
    value = _VALUE_ALIASES.get(value, value)
    if value not in LABEL_CATALOG.get(field, ()):
        return None
    return f"{field}:{value}"


def normalize_labels(raw_labels: object) -> list[str]:
    if not isinstance(raw_labels, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_label in raw_labels:
        label = normalize_label(raw_label)
        if label and label not in seen:
            normalized.append(label)
            seen.add(label)
    return _drop_conflicting_labels(normalized)


def _normalize_token(value: str) -> str:
    return "_".join(value.strip().lower().replace("-", "_").split())


def _drop_conflicting_labels(labels: list[str]) -> list[str]:
    result: list[str] = []
    selected_by_field: dict[str, str] = {}
    for label in labels:
        field, _, _value = label.partition(":")
        if field in _EXCLUSIVE_FIELDS and field in selected_by_field:
            continue
        if field in _EXCLUSIVE_FIELDS:
            selected_by_field[field] = label
        result.append(label)
    return result
