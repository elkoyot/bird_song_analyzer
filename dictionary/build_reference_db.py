#!/usr/bin/env python3
"""
Build reference.db — ready-made SQLite for the Bird Song Analyzer app.

Reads:
  - bird/bird_data_collector/output/summary.csv      → taxon_order, taxon_family, species
  - bird/bird_data_collector/output/synonyms.csv      → taxonomy_synonym
  - bird/bird_data_collector/output/names_all_languages.csv → species_name
  - countries_bird/*.json                             → species_country
  - ../app/src/main/assets/geo/seed.json              → geo_entity, ml_model, geo_model, translation
  - ../app/src/main/assets/birdnet/v24/labels/en_us.txt → model_map_v24.csv
  - ../app/src/main/assets/birdnet/v30/labels.csv       → model_map_v30.csv

Writes:
  - ../app/src/main/assets/db/reference.db
  - ../app/src/main/assets/birdnet/v24/model_map.csv
  - ../app/src/main/assets/birdnet/v30/model_map.csv
  - report.txt
"""

import csv
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ASSETS_DIR = PROJECT_ROOT / "app" / "src" / "main" / "assets"

SUMMARY_CSV = SCRIPT_DIR / "bird" / "bird_data_collector" / "output" / "summary.csv"
SYNONYMS_CSV = SCRIPT_DIR / "bird" / "bird_data_collector" / "output" / "synonyms.csv"
NAMES_CSV = SCRIPT_DIR / "bird" / "bird_data_collector" / "output" / "names_all_languages.csv"
COUNTRIES_BIRD_DIR = SCRIPT_DIR / "countries_bird"

SEED_JSON = ASSETS_DIR / "geo" / "seed.json"
V24_LABELS = ASSETS_DIR / "birdnet" / "v24" / "labels" / "en_us.txt"
V30_LABELS = ASSETS_DIR / "birdnet" / "v30" / "labels.csv"

OUT_DB = ASSETS_DIR / "db" / "reference.db"
OUT_MAP_V24 = ASSETS_DIR / "birdnet" / "v24" / "model_map.csv"
OUT_MAP_V30 = ASSETS_DIR / "birdnet" / "v30" / "model_map.csv"
OUT_REPORT = SCRIPT_DIR / "report.txt"

# Languages to include in species_name from names_all_languages.csv
LANGUAGES = {"Английский"}
LANG_MAP = {"Английский": "en"}

# Regex to extract Latin binomial from dirty page titles
# Matches: "Capitalized word" + "lowercase word" (optionally followed by junk)
LATIN_BINOMIAL_RE = re.compile(r'\b([A-Z][a-z]+)\s+([a-z]+)\b')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
report_lines: list[str] = []


def report(msg: str) -> None:
    report_lines.append(msg)
    print(msg)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Read summary.csv → orders, families, species
# ---------------------------------------------------------------------------
def clean_latin_name(raw: str) -> str | None:
    """Extract clean Latin binomial from potentially dirty page titles.

    Examples:
      "Parus major" → "Parus major"
      "Коростель — Crex crex: полный обзор" → "Crex crex"
      "Зарянка (Erithacus rubecula) — описание" → "Erithacus rubecula"
    """
    raw = raw.strip()
    if not raw:
        return None
    # Already clean: starts with uppercase Latin letter, second word lowercase
    if re.match(r'^[A-Z][a-z]+ [a-z]+', raw):
        # Take first two words only
        parts = raw.split()
        return f"{parts[0]} {parts[1]}"
    # Dirty: try to find Latin binomial inside
    m = LATIN_BINOMIAL_RE.search(raw)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def read_summary() -> tuple[
    dict[str, dict],   # orders:   latinName → {taxonClass, nameRu, nameEn}
    dict[str, dict],   # families: latinName → {orderLat, nameRu, nameEn}
    list[dict],         # species list
    list[dict],         # russian names from summary (name_ru column)
]:
    orders: dict[str, dict] = {}
    families: dict[str, dict] = {}
    species_list: list[dict] = []
    ru_names: list[dict] = []
    dirty_count = 0

    with open(SUMMARY_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row["name_lat"].strip()
            sci_name = clean_latin_name(raw_name)
            if not sci_name:
                continue
            if sci_name != raw_name.strip():
                dirty_count += 1
                # Extract Russian name from dirty title if name_ru is empty
                # "Коростель — Crex crex: ..." → "Коростель"
                # "Зарянка (Erithacus rubecula) — ..." → "Зарянка"
                name_ru_raw = row.get("name_ru", "").strip()
                if not name_ru_raw:
                    # Try to get Russian part before — or (
                    title_part = re.split(r'\s*[—(]', raw_name)[0].strip()
                    if title_part and re.match(r'^[А-Яа-яЁё]', title_part):
                        row["name_ru"] = title_part

            order_lat = row.get("order_lat", "").strip()
            order_ru = row.get("order_ru", "").strip()
            family_lat = row.get("family_lat", "").strip()
            family_ru = row.get("family_ru", "").strip()
            genus = row.get("genus", "").strip()
            iucn = row.get("iucn_status", "").strip() or None

            # Collect orders (taxonClass = "Aves" for all dibird data)
            if order_lat and order_lat not in orders:
                orders[order_lat] = {
                    "taxonClass": "Aves",
                    "nameRu": order_ru,
                    "nameEn": order_lat,  # English name = Latin for orders
                }

            # Collect families
            if family_lat and family_lat not in families:
                families[family_lat] = {
                    "orderLat": order_lat,
                    "nameRu": family_ru,
                    "nameEn": family_lat,  # English name = Latin for families
                }

            species_list.append({
                "scientificName": sci_name,
                "familyLat": family_lat,
                "genus": genus,
                "iucnStatus": iucn,
            })

            # Russian name from summary (name_ru column)
            name_ru = row.get("name_ru", "").strip()
            if name_ru:
                ru_names.append({
                    "scientificName": sci_name,
                    "lang": "ru",
                    "name": name_ru,
                })

    report(f"Summary: {len(species_list)} species, {len(orders)} orders, {len(families)} families")
    if dirty_count:
        report(f"  Cleaned dirty name_lat: {dirty_count}")
    report(f"  Russian names from summary: {len(ru_names)}")
    return orders, families, species_list, ru_names


# ---------------------------------------------------------------------------
# Step 2: Read synonyms.csv
# ---------------------------------------------------------------------------
def read_synonyms() -> list[dict]:
    synonyms: list[dict] = []
    with open(SYNONYMS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sci_name = clean_latin_name(row["name_lat"].strip())
            syn_type = row["type"].strip()
            synonym = row["synonym"].strip()
            if sci_name and synonym:
                synonyms.append({
                    "synonym": synonym,
                    "scientificName": sci_name,
                    "type": syn_type,
                })
    report(f"Synonyms: {len(synonyms)} entries")
    return synonyms


# ---------------------------------------------------------------------------
# Step 3: Read names_all_languages.csv → species_name (filtered)
# ---------------------------------------------------------------------------
def read_names() -> list[dict]:
    names: list[dict] = []
    with open(NAMES_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang_full = row["language"].strip()
            if lang_full not in LANGUAGES:
                continue
            raw_name = row["name_lat"].strip()
            sci_name = clean_latin_name(raw_name)
            raw_names = row["names"].strip()
            if not sci_name or not raw_names:
                continue
            # Take first name as primary
            primary_name = raw_names.split(",")[0].strip()
            if primary_name:
                names.append({
                    "scientificName": sci_name,
                    "lang": LANG_MAP[lang_full],
                    "name": primary_name,
                })
    report(f"Species names (en): {len(names)} entries")
    return names


# ---------------------------------------------------------------------------
# Step 4: Read countries_bird/*.json → species_country
# ---------------------------------------------------------------------------
def read_species_countries() -> list[dict]:
    links: list[dict] = []
    seen = set()
    for path in sorted(COUNTRIES_BIRD_DIR.glob("*.json")):
        # Extract country code from filename: "RU_Россия_Russia.json" → "RU"
        code = path.stem.split("_")[0]
        with open(path, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            continue
        data = json.loads(content)
        birds = data.get("birds", [])
        for bird in birds:
            sci_name = bird.get("name_lat", "").strip()
            if sci_name:
                key = (sci_name, code)
                if key not in seen:
                    seen.add(key)
                    links.append({
                        "scientificName": sci_name,
                        "countryCode": code,
                    })
    report(f"Species-country links: {len(links)} entries from {len(list(COUNTRIES_BIRD_DIR.glob('*.json')))} countries")
    return links


# ---------------------------------------------------------------------------
# Step 5: Read seed.json → geo data + translations
# ---------------------------------------------------------------------------
def read_seed() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Returns (geo_entities, ml_models, geo_models, translations)."""
    with open(SEED_JSON, encoding="utf-8") as f:
        seed = json.load(f)

    geo_entities: list[dict] = []
    translations: list[dict] = []
    ml_models: list[dict] = []
    geo_models: list[dict] = []

    for idx, continent in enumerate(seed["continents"]):
        code = continent["code"]
        name_ru = continent["nameRu"]
        name_en = continent["nameEn"]

        geo_entities.append({
            "code": code, "type": "continent", "parent_code": None,
            "name_ru": name_ru, "name_en": name_en,
            "min_lat": None, "max_lat": None,
            "min_lon": None, "max_lon": None,
            "buffer_deg": 2.5, "sort_order": idx,
        })

        # Continent translation
        translations.append({"entityType": "continent", "entityKey": code, "lang": "ru", "name": name_ru})
        translations.append({"entityType": "continent", "entityKey": code, "lang": "en", "name": name_en})

        for country in continent.get("countries", []):
            cc = country["code"]
            bbox = country.get("bbox", {})
            buf = country.get("bufferDeg", 2.5)

            geo_entities.append({
                "code": cc, "type": "country", "parent_code": code,
                "name_ru": country.get("nameRu", cc),
                "name_en": country.get("nameEn", cc),
                "min_lat": bbox.get("minLat"), "max_lat": bbox.get("maxLat"),
                "min_lon": bbox.get("minLon"), "max_lon": bbox.get("maxLon"),
                "buffer_deg": buf, "sort_order": 0,
            })

            for region in country.get("regions", []):
                rc = region["code"]
                rbbox = region.get("bbox", {})
                geo_entities.append({
                    "code": rc, "type": "region", "parent_code": cc,
                    "name_ru": region.get("nameRu", rc),
                    "name_en": region.get("nameEn", rc),
                    "min_lat": rbbox.get("minLat"), "max_lat": rbbox.get("maxLat"),
                    "min_lon": rbbox.get("minLon"), "max_lon": rbbox.get("maxLon"),
                    "buffer_deg": buf, "sort_order": 0,
                })

                # Region translation
                r_name_ru = region.get("nameRu", rc)
                r_name_en = region.get("nameEn", rc)
                translations.append({"entityType": "region", "entityKey": rc, "lang": "ru", "name": r_name_ru})
                translations.append({"entityType": "region", "entityKey": rc, "lang": "en", "name": r_name_en})

    for m in seed.get("models", []):
        ml_models.append({
            "id": m["id"], "name": m["name"], "runtime": m["runtime"],
            "audio_path": m["audioPath"], "meta_model_path": m.get("metaModelPath"),
            "labels_path": m["labelsPath"], "sample_rate": m["sampleRate"],
            "chunk_seconds": m["chunkSeconds"], "species_count": m["speciesCount"],
            "is_bundled": m["isBundled"],
        })

    for gm in seed.get("geoModels", []):
        geo_models.append({
            "geo_code": gm["geoCode"],
            "model_id": gm["modelId"],
            "is_default": gm.get("isDefault", False),
        })

    report(f"Geo: {len(geo_entities)} entities, {len(ml_models)} models, {len(geo_models)} geo-model links, {len(translations)} translations")
    return geo_entities, ml_models, geo_models, translations


# ---------------------------------------------------------------------------
# Step 6: Read model labels → build model_map CSVs
# ---------------------------------------------------------------------------
def read_v24_labels() -> list[tuple[str, str]]:
    """Returns list of (scientificName, commonName) by labelIndex."""
    labels = []
    with open(V24_LABELS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("_", 1)
            sci = parts[0].strip()
            common = parts[1].strip() if len(parts) > 1 else ""
            labels.append((sci, common))
    return labels


def read_v30_labels() -> list[dict]:
    """Returns list of {id, sci_name, com_name, gbif, class, order} by labelIndex."""
    labels = []
    with open(V30_LABELS, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            labels.append({
                "sci_name": row["sci_name"].strip(),
                "com_name": row["com_name"].strip(),
                "taxon_class": row["class"].strip(),
                "order": row.get("order", "").strip(),
            })
    return labels


def build_model_maps(
    v24_labels: list[tuple[str, str]],
    v30_labels: list[dict],
    species_set: set[str],
    synonym_map: dict[str, str],  # old_latin → current scientificName
) -> tuple[list[dict], list[dict]]:
    """Build model maps, resolving synonyms. Returns (v24_map, v30_map)."""

    def resolve(model_label: str) -> str | None:
        """Resolve model label to reference scientificName."""
        if model_label in species_set:
            return model_label
        if model_label in synonym_map:
            return synonym_map[model_label]
        return None

    v24_map = []
    v24_resolved = 0
    v24_synonym = 0
    v24_missing = []
    for idx, (sci, common) in enumerate(v24_labels):
        resolved = resolve(sci)
        if resolved:
            if resolved != sci:
                v24_synonym += 1
            v24_resolved += 1
            v24_map.append({
                "labelIndex": idx,
                "modelLabel": sci,
                "scientificName": resolved,
                "taxonClass": "Aves",  # V2.4 has no class field, assume Aves
            })
        else:
            v24_missing.append(sci)
            v24_map.append({
                "labelIndex": idx,
                "modelLabel": sci,
                "scientificName": "",
                "taxonClass": "",
            })

    v30_map = []
    v30_resolved = 0
    v30_synonym = 0
    v30_missing = []
    for idx, lbl in enumerate(v30_labels):
        sci = lbl["sci_name"]
        resolved = resolve(sci)
        taxon = lbl["taxon_class"]
        if resolved:
            if resolved != sci:
                v30_synonym += 1
            v30_resolved += 1
            v30_map.append({
                "labelIndex": idx,
                "modelLabel": sci,
                "scientificName": resolved,
                "taxonClass": taxon,
            })
        else:
            v30_missing.append(f"{sci} ({taxon})")
            v30_map.append({
                "labelIndex": idx,
                "modelLabel": sci,
                "scientificName": "",
                "taxonClass": taxon,
            })

    report(f"\nModel Map V2.4: {len(v24_labels)} labels total")
    report(f"  Resolved: {v24_resolved} ({v24_synonym} via synonym)")
    report(f"  Missing from reference: {len(v24_missing)}")
    if v24_missing[:20]:
        for m in v24_missing[:20]:
            report(f"    - {m}")
        if len(v24_missing) > 20:
            report(f"    ... and {len(v24_missing) - 20} more")

    report(f"\nModel Map V3.0: {len(v30_labels)} labels total")
    report(f"  Resolved: {v30_resolved} ({v30_synonym} via synonym)")
    report(f"  Missing from reference: {len(v30_missing)}")
    if v30_missing[:20]:
        for m in v30_missing[:20]:
            report(f"    - {m}")
        if len(v30_missing) > 20:
            report(f"    ... and {len(v30_missing) - 20} more")

    return v24_map, v30_map


# ---------------------------------------------------------------------------
# Step 7: Add non-bird species from V3.0 to reference
# ---------------------------------------------------------------------------
def add_non_bird_species(
    v30_labels: list[dict],
    orders: dict[str, dict],
    families: dict[str, dict],
    species_list: list[dict],
    species_set: set[str],
    names_list: list[dict],
) -> None:
    """Add Insecta/Mammalia/Amphibia/Squamata species from V3.0 labels
    that are not already in the reference."""
    added = 0
    for lbl in v30_labels:
        sci = lbl["sci_name"]
        taxon = lbl["taxon_class"]
        order_lat = lbl.get("order", "")

        if taxon == "Aves" or sci in species_set:
            continue

        # Add order if new
        if order_lat and order_lat not in orders:
            orders[order_lat] = {
                "taxonClass": taxon,
                "nameRu": order_lat,
                "nameEn": order_lat,
            }

        # Genus from scientific name
        genus = sci.split()[0] if " " in sci else sci

        # We don't have family info for non-birds from V3.0, use placeholder
        family_lat = f"_{taxon}_unknown"
        if family_lat not in families:
            families[family_lat] = {
                "orderLat": order_lat,
                "nameRu": taxon,
                "nameEn": taxon,
            }

        species_list.append({
            "scientificName": sci,
            "familyLat": family_lat,
            "genus": genus,
            "iucnStatus": None,
        })
        species_set.add(sci)

        # Add English name from V3.0 label
        com_name = lbl["com_name"]
        if com_name:
            names_list.append({
                "scientificName": sci,
                "lang": "en",
                "name": com_name,
            })

        added += 1

    report(f"Non-bird species added from V3.0: {added}")


# ---------------------------------------------------------------------------
# Step 8: Build SQLite database
# ---------------------------------------------------------------------------
def build_db(
    orders: dict[str, dict],
    families: dict[str, dict],
    species_list: list[dict],
    names_list: list[dict],
    synonyms: list[dict],
    species_countries: list[dict],
    geo_entities: list[dict],
    ml_models: list[dict],
    geo_models: list[dict],
    translations: list[dict],
    order_translations: list[dict],
    family_translations: list[dict],
) -> None:
    ensure_dir(OUT_DB)
    if OUT_DB.exists():
        OUT_DB.unlink()

    conn = sqlite3.connect(str(OUT_DB))
    c = conn.cursor()

    # Enable WAL mode for better read performance
    c.execute("PRAGMA journal_mode=WAL")

    # ── Create tables ──

    # ── DDL must match Room-generated schema exactly ──
    # Rules:
    # - PK columns must have NOT NULL
    # - No DEFAULT values (Kotlin defaults ≠ SQL defaults for Room)
    # - Index names: index_{tableName}_{columnName}
    # - ForeignKey ON DELETE must match Entity annotations

    c.execute("""
        CREATE TABLE taxon_order (
            id          INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            latin_name  TEXT NOT NULL,
            taxon_class TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE taxon_family (
            id          INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            latin_name  TEXT NOT NULL,
            order_id    INTEGER NOT NULL,
            FOREIGN KEY (order_id) REFERENCES taxon_order(id) ON DELETE CASCADE
        )
    """)
    c.execute("CREATE INDEX index_taxon_family_order_id ON taxon_family(order_id)")

    c.execute("""
        CREATE TABLE species (
            scientific_name TEXT NOT NULL PRIMARY KEY,
            family_id       INTEGER,
            genus           TEXT,
            iucn_status     TEXT,
            FOREIGN KEY (family_id) REFERENCES taxon_family(id) ON DELETE SET NULL
        )
    """)
    c.execute("CREATE INDEX index_species_family_id ON species(family_id)")
    c.execute("CREATE INDEX index_species_genus ON species(genus)")

    c.execute("""
        CREATE TABLE species_name (
            scientific_name TEXT NOT NULL,
            lang            TEXT NOT NULL,
            name            TEXT NOT NULL,
            PRIMARY KEY (scientific_name, lang),
            FOREIGN KEY (scientific_name) REFERENCES species(scientific_name) ON DELETE CASCADE
        )
    """)
    c.execute("CREATE INDEX index_species_name_name ON species_name(name)")
    c.execute("CREATE INDEX index_species_name_lang ON species_name(lang)")

    c.execute("""
        CREATE TABLE taxonomy_synonym (
            synonym         TEXT NOT NULL PRIMARY KEY,
            scientific_name TEXT NOT NULL,
            type            TEXT NOT NULL,
            FOREIGN KEY (scientific_name) REFERENCES species(scientific_name) ON DELETE CASCADE
        )
    """)

    c.execute("""
        CREATE TABLE species_country (
            scientific_name TEXT NOT NULL,
            country_code    TEXT NOT NULL,
            PRIMARY KEY (scientific_name, country_code),
            FOREIGN KEY (scientific_name) REFERENCES species(scientific_name) ON DELETE CASCADE
        )
    """)
    c.execute("CREATE INDEX index_species_country_country_code ON species_country(country_code)")

    c.execute("""
        CREATE TABLE translation (
            entity_type TEXT NOT NULL,
            entity_key  TEXT NOT NULL,
            lang        TEXT NOT NULL,
            name        TEXT NOT NULL,
            PRIMARY KEY (entity_type, entity_key, lang)
        )
    """)

    # ── Geo tables (must match Room Entity annotations exactly) ──

    c.execute("""
        CREATE TABLE geo_entity (
            code        TEXT NOT NULL PRIMARY KEY,
            type        TEXT NOT NULL,
            parent_code TEXT,
            name_ru     TEXT NOT NULL,
            name_en     TEXT NOT NULL,
            min_lat     REAL,
            max_lat     REAL,
            min_lon     REAL,
            max_lon     REAL,
            buffer_deg  REAL NOT NULL,
            sort_order  INTEGER NOT NULL,
            FOREIGN KEY (parent_code) REFERENCES geo_entity(code) ON DELETE CASCADE
        )
    """)
    c.execute("CREATE INDEX index_geo_entity_parent_code ON geo_entity(parent_code)")
    c.execute("CREATE INDEX index_geo_entity_type ON geo_entity(type)")

    c.execute("""
        CREATE TABLE ml_model (
            id              TEXT NOT NULL PRIMARY KEY,
            name            TEXT NOT NULL,
            runtime         TEXT NOT NULL,
            audio_path      TEXT NOT NULL,
            meta_model_path TEXT,
            labels_path     TEXT NOT NULL,
            sample_rate     INTEGER NOT NULL,
            chunk_seconds   INTEGER NOT NULL,
            species_count   INTEGER NOT NULL,
            is_bundled      INTEGER NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE geo_model (
            geo_code    TEXT NOT NULL,
            model_id    TEXT NOT NULL,
            is_default  INTEGER NOT NULL,
            PRIMARY KEY (geo_code, model_id),
            FOREIGN KEY (geo_code) REFERENCES geo_entity(code) ON DELETE CASCADE,
            FOREIGN KEY (model_id) REFERENCES ml_model(id) ON DELETE CASCADE
        )
    """)

    # ── Insert data ──

    # Orders
    order_id_map: dict[str, int] = {}
    for latin, info in orders.items():
        c.execute(
            "INSERT INTO taxon_order (latin_name, taxon_class) VALUES (?, ?)",
            (latin, info["taxonClass"]),
        )
        order_id_map[latin] = c.lastrowid  # type: ignore

    # Families
    family_id_map: dict[str, int] = {}
    for latin, info in families.items():
        oid = order_id_map.get(info["orderLat"])
        if oid is None:
            continue
        c.execute(
            "INSERT INTO taxon_family (latin_name, order_id) VALUES (?, ?)",
            (latin, oid),
        )
        family_id_map[latin] = c.lastrowid  # type: ignore

    # Species
    species_set_inserted: set[str] = set()
    for sp in species_list:
        sci = sp["scientificName"]
        if sci in species_set_inserted:
            continue
        fid = family_id_map.get(sp["familyLat"])
        c.execute(
            "INSERT INTO species (scientific_name, family_id, genus, iucn_status) VALUES (?, ?, ?, ?)",
            (sci, fid, sp["genus"], sp["iucnStatus"]),
        )
        species_set_inserted.add(sci)

    # Species names
    seen_names: set[tuple[str, str]] = set()
    for sn in names_list:
        key = (sn["scientificName"], sn["lang"])
        if key in seen_names:
            continue
        if sn["scientificName"] not in species_set_inserted:
            continue
        seen_names.add(key)
        c.execute(
            "INSERT INTO species_name (scientific_name, lang, name) VALUES (?, ?, ?)",
            (sn["scientificName"], sn["lang"], sn["name"]),
        )

    # Synonyms (only old_latin that reference existing species)
    syn_inserted = 0
    syn_skipped = 0
    seen_synonyms: set[str] = set()
    for syn in synonyms:
        if syn["scientificName"] not in species_set_inserted:
            syn_skipped += 1
            continue
        if syn["synonym"] in seen_synonyms:
            continue
        # Don't add synonym if it's the same as the current name
        if syn["synonym"] == syn["scientificName"]:
            continue
        seen_synonyms.add(syn["synonym"])
        c.execute(
            "INSERT INTO taxonomy_synonym (synonym, scientific_name, type) VALUES (?, ?, ?)",
            (syn["synonym"], syn["scientificName"], syn["type"]),
        )
        syn_inserted += 1
    report(f"Synonyms inserted: {syn_inserted}, skipped (species not found): {syn_skipped}")

    # Species-country
    sc_inserted = 0
    sc_skipped = 0
    for sc in species_countries:
        if sc["scientificName"] not in species_set_inserted:
            sc_skipped += 1
            continue
        c.execute(
            "INSERT OR IGNORE INTO species_country (scientific_name, country_code) VALUES (?, ?)",
            (sc["scientificName"], sc["countryCode"]),
        )
        sc_inserted += 1
    report(f"Species-country: inserted {sc_inserted}, skipped {sc_skipped}")

    # Translations (continents, regions)
    for t in translations:
        c.execute(
            "INSERT OR IGNORE INTO translation (entity_type, entity_key, lang, name) VALUES (?, ?, ?, ?)",
            (t["entityType"], t["entityKey"], t["lang"], t["name"]),
        )

    # Translations (orders, families)
    for t in order_translations:
        c.execute(
            "INSERT OR IGNORE INTO translation (entity_type, entity_key, lang, name) VALUES (?, ?, ?, ?)",
            (t["entityType"], t["entityKey"], t["lang"], t["name"]),
        )
    for t in family_translations:
        c.execute(
            "INSERT OR IGNORE INTO translation (entity_type, entity_key, lang, name) VALUES (?, ?, ?, ?)",
            (t["entityType"], t["entityKey"], t["lang"], t["name"]),
        )

    # Geo entities (insert continents first, then countries, then regions for FK)
    for ge in [g for g in geo_entities if g["type"] == "continent"]:
        c.execute(
            """INSERT INTO geo_entity
               (code, type, parent_code, name_ru, name_en, min_lat, max_lat, min_lon, max_lon, buffer_deg, sort_order)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ge["code"], ge["type"], ge["parent_code"], ge["name_ru"], ge["name_en"],
             ge["min_lat"], ge["max_lat"], ge["min_lon"], ge["max_lon"], ge["buffer_deg"], ge["sort_order"]),
        )
    for ge in [g for g in geo_entities if g["type"] == "country"]:
        c.execute(
            """INSERT INTO geo_entity
               (code, type, parent_code, name_ru, name_en, min_lat, max_lat, min_lon, max_lon, buffer_deg, sort_order)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ge["code"], ge["type"], ge["parent_code"], ge["name_ru"], ge["name_en"],
             ge["min_lat"], ge["max_lat"], ge["min_lon"], ge["max_lon"], ge["buffer_deg"], ge["sort_order"]),
        )
    for ge in [g for g in geo_entities if g["type"] == "region"]:
        c.execute(
            """INSERT INTO geo_entity
               (code, type, parent_code, name_ru, name_en, min_lat, max_lat, min_lon, max_lon, buffer_deg, sort_order)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ge["code"], ge["type"], ge["parent_code"], ge["name_ru"], ge["name_en"],
             ge["min_lat"], ge["max_lat"], ge["min_lon"], ge["max_lon"], ge["buffer_deg"], ge["sort_order"]),
        )

    # ML models
    for m in ml_models:
        c.execute(
            """INSERT INTO ml_model
               (id, name, runtime, audio_path, meta_model_path, labels_path, sample_rate, chunk_seconds, species_count, is_bundled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (m["id"], m["name"], m["runtime"], m["audio_path"], m["meta_model_path"],
             m["labels_path"], m["sample_rate"], m["chunk_seconds"], m["species_count"], m["is_bundled"]),
        )

    # Geo-model mappings
    for gm in geo_models:
        c.execute(
            "INSERT INTO geo_model (geo_code, model_id, is_default) VALUES (?, ?, ?)",
            (gm["geo_code"], gm["model_id"], int(gm["is_default"])),
        )

    conn.commit()

    # Stats
    c.execute("SELECT COUNT(*) FROM taxon_order")
    report(f"\n=== DB Stats ===")
    report(f"taxon_order:     {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM taxon_family")
    report(f"taxon_family:    {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM species")
    report(f"species:         {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM species_name")
    report(f"species_name:    {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM taxonomy_synonym")
    report(f"taxonomy_synonym:{c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM species_country")
    report(f"species_country: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM translation")
    report(f"translation:     {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM geo_entity")
    report(f"geo_entity:      {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM ml_model")
    report(f"ml_model:        {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM geo_model")
    report(f"geo_model:       {c.fetchone()[0]}")

    # Optimize
    c.execute("VACUUM")
    conn.commit()
    conn.close()

    size_mb = OUT_DB.stat().st_size / (1024 * 1024)
    report(f"\nDatabase size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Step 9: Write model_map CSVs
# ---------------------------------------------------------------------------
def write_model_maps(v24_map: list[dict], v30_map: list[dict]) -> None:
    ensure_dir(OUT_MAP_V24)
    with open(OUT_MAP_V24, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["labelIndex", "modelLabel", "scientificName", "taxonClass"])
        w.writeheader()
        w.writerows(v24_map)
    report(f"Model map V2.4: {OUT_MAP_V24} ({len(v24_map)} entries)")

    ensure_dir(OUT_MAP_V30)
    with open(OUT_MAP_V30, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["labelIndex", "modelLabel", "scientificName", "taxonClass"])
        w.writeheader()
        w.writerows(v30_map)
    report(f"Model map V3.0: {OUT_MAP_V30} ({len(v30_map)} entries)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    report("=" * 60)
    report("Building reference.db")
    report("=" * 60)

    # 1. Read all sources
    orders, families, species_list, ru_names = read_summary()
    synonyms = read_synonyms()
    names_list = read_names()
    # Add Russian names from summary.csv (not in names_all_languages.csv)
    names_list.extend(ru_names)
    species_countries = read_species_countries()
    geo_entities, ml_models, geo_models, geo_translations = read_seed()

    # Species set for lookups
    species_set = {sp["scientificName"] for sp in species_list}

    # 2. Read model labels
    v24_labels = read_v24_labels()
    v30_labels = read_v30_labels()

    # 3. Add non-bird species from V3.0 to reference
    add_non_bird_species(v30_labels, orders, families, species_list, species_set, names_list)

    # 4. Build synonym lookup (old_latin only)
    synonym_map: dict[str, str] = {}
    for syn in synonyms:
        if syn["type"] == "old_latin":
            synonym_map[syn["synonym"]] = syn["scientificName"]

    # 5. Build model maps
    v24_map, v30_map = build_model_maps(v24_labels, v30_labels, species_set, synonym_map)

    # 6. Build order/family translations
    order_translations = []
    for latin, info in orders.items():
        order_translations.append({"entityType": "order", "entityKey": latin, "lang": "ru", "name": info["nameRu"]})
        order_translations.append({"entityType": "order", "entityKey": latin, "lang": "en", "name": info["nameEn"]})

    family_translations = []
    for latin, info in families.items():
        family_translations.append({"entityType": "family", "entityKey": latin, "lang": "ru", "name": info["nameRu"]})
        family_translations.append({"entityType": "family", "entityKey": latin, "lang": "en", "name": info["nameEn"]})

    # 7. Build SQLite
    build_db(
        orders, families, species_list, names_list, synonyms, species_countries,
        geo_entities, ml_models, geo_models, geo_translations,
        order_translations, family_translations,
    )

    # 8. Write model maps
    write_model_maps(v24_map, v30_map)

    # 9. Additional stats for report
    report(f"\n=== Species without Russian name ===")
    species_with_ru = {sn["scientificName"] for sn in names_list if sn["lang"] == "ru"}
    no_ru = [sp["scientificName"] for sp in species_list if sp["scientificName"] not in species_with_ru]
    report(f"Total: {len(no_ru)}")
    if no_ru[:10]:
        for n in no_ru[:10]:
            report(f"  - {n}")
        if len(no_ru) > 10:
            report(f"  ... and {len(no_ru) - 10} more")

    # 10. Save report
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {OUT_REPORT}")
    print(f"Database saved to {OUT_DB}")


if __name__ == "__main__":
    main()
