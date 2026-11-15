"""Materials specialist: what things are made of and where each material is used, from whatever the drawing holds.

Questions like "which materials are used in the exterior walls?", "where is concrete used?", "what is the floor finish
in room 113?", "how many walls are brick?", "total volume of concrete in the slabs?" are answered from, in order:

  1. the BIM model's own material links (IfcRelAssociatesMaterial: layers with thicknesses, constituents, profiles),
     or the element type names when the model links no material ("Basiswand:STB 250" = reinforced concrete)
                                                                   -> stoa_qa.materials_ifc       direct
  2. the sheet's printed words: finish / door schedules and their finish keys (followed across the drawing set to
     the room tags on a plan), material notes and labels (CONC., GYP. BD., "PARKETTI"), legends and keynotes
                                                                   -> stoa_qa.materials_sheet     direct
  3. hatches drawn on the plan: CAD hatch names (DXF), or hatch textures recognised on the sheet image by the trained
     hatch classifier, read through the sheet's own legend when it has one, else through drafting convention
                                                                   -> stoa_qa.hatch               inferred
  4. the vision reader (Gemma reads the sheet; every material it names must come with a box)       review
and only when none of these finds anything does the answer say so, listing what was searched.

No LLM is used for 1-3: material names are matched with a multilingual vocabulary (FAMILIES) that also knows the
drafting abbreviations of US and European sheets. Numbers (counts, volumes, thicknesses) are computed by our code.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .core import Answer, Evidence


def fold(s: str) -> str:
    """Lower case without accents (béton -> beton, Fußboden -> fussboden), for matching across languages."""
    s = s.replace("ß", "ss").replace("ø", "o").replace("æ", "ae").replace("ł", "l").replace("Ł", "l")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


# ------------------------------------------------------------------ vocabulary
# family: (English name, word stems, drafting abbreviations).
# A stem matches at the start of a word ("tile" -> tiles, tiling, not textile); a stem starting with "*" also matches
# inside compounds (German, Nordic, Dutch: Stahl*beton*, Kalksand*stein*, *gips*karton). Abbreviations match only as
# printed upper-case tokens, alone or as a tag (CONC., GWB, WD-1, AL-2/GL-2).
FAMILIES: dict[str, tuple[str, list[str], list[str]]] = {
    "concrete": ("concrete", ["concrete", "*beton", "*betong", "*betoni", "calcestruzz", "cemento armato", "hormig", "*betonu",
                              "sichtbeton", "ortbeton", "stahlbeton", "leichtbeton", "gasbeton", "porenbeton", "cast-in-place",
                              "precast", "reinforced concrete", "shotcrete", "slab on grade"],
                 ["CONC", "CONCR", "RC", "STB", "STBT", "BET"]),
    "masonry": ("masonry", ["masonry", "*mauerwerk", "maconnerie", "muratura", "metselwerk", "*murverk", "mur de", "mampost",
                            "zidarie", "murowan", "*muuraus", "kalksandstein", "sand-lime", "kalkzandsteen", "ytong", "gas concrete"],
                ["MAS", "MASY", "KS", "MW"]),
    "brick": ("brick", ["brick", "*ziegel", "brique", "mattone", "mattoni", "baksteen", "*tegl", "*tegel", "*tiili", "ladrill",
                        "cegl", "caramid", "*klinker", "*backstein", "telles"], ["BRK", "BR"]),
    "block": ("concrete block", ["concrete block", "cmu", "cinder block", "breeze block", "*betonblock",
                                 "*betonstein", "parpaing", "blocco", "betonblok", "*harkko", "bloque", "pustak"], ["CMU", "CB"]),
    "stone": ("stone", ["stone", "*stein", "pierre", "pietra", "*steen", "sten", "*kivi", "piedra", "kamien", "piatra",
                        "granite", "*granit", "marble", "*marmor", "marbre", "marmo", "slate", "schiefer", "limestone", "kalkstein",
                        "sandstone", "sandstein", "travertine", "terrazzo", "quartz"], ["STN", "GRAN", "MRB"]),
    "wood": ("wood", ["wood", "wooden", "timber", "lumber", "*holz", "bois", "legno", "legname", "*trae", "*puu",
                      "madera", "drewn", "lemn", "hardwood", "softwood", "glulam", "clt", "cross-laminated", "*brettsperr",
                      "*brettschicht", "*kertopuu", "oak", "chene", "rovere", "tammi", "pine", "kiefer",
                      "*fichte", "spruce", "birch", "*koivu", "beech", "buche", "larch", "walnut", "*lauta", "*naaldhout", "*hardhout", "hout",
                      "stud wall", "timber frame", "*holzständer", "hsb"],
             ["WD", "WOOD", "HW", "LVL", "GLB"]),
    "plywood": ("plywood / wood-based board", ["plywood", "osb", "particle board", "particleboard", "chipboard", "mdf", "hdf",
                                               "sperrholz", "*spanplatte", "contreplaque", "compensato", "multiplex", "*vaneri",
                                               "*lastulevy", "sheathing"], ["PLY", "PLYWD", "OSB", "MDF"]),
    "gypsum": ("gypsum board / plasterboard", ["gypsum", "plasterboard", "drywall", "gyp", "sheetrock", "wallboard", "*gips",
                                               "*gipskarton", "rigips", "trockenbau", "placo", "platre", "cartongesso", "*gipsplaat",
                                               "*gipsplade", "*gipsplate", "*kipsi", "pladur", "yeso", "regips", "karton-gips"],
               ["GWB", "GYP", "GB", "GKB", "GKF", "GK", "GYPBD", "GYP.BD"]),
    "plaster": ("plaster / render", ["plaster", "render", "stucco", "*putz", "enduit", "intonaco", "*pleister", "puts", "*laasti",
                                     "rappatura", "enlucido", "tynk", "tencuial", "slamning"], ["PLAS", "PLST"]),
    "steel": ("steel", ["steel", "*stahl", "acier", "acciaio", "*staal", "teras", "acero", "otel", "stainless",
                        "*edelstahl", "*baustahl", "inox", "hollow metal", "galvaniz", "w-shape", "wide flange"],
              ["STL", "ST", "SS", "HM", "GALV"]),
    "aluminium": ("aluminium", ["aluminium", "aluminum", "alu", "allumin", "*alumiini", "alumin"], ["AL", "ALUM", "ALU"]),
    "metal": ("metal", ["metal", "*metall", "metaal", "metallo", "*metalli", "zinc", "*zink", "zinco", "copper", "*kupfer",
                        "cuivre", "koper", "brass", "bronze", "titanzink", "titanium", "sheet metal",
                        "corrugated"], ["MTL", "MET", "ZN", "CU"]),
    "glass": ("glass", ["glass", "glazing", "glazed", "*glas", "verre", "vetro", "*lasi", "vidrio", "szkl", "sticla", "*verglasung",
                        "*glazen", "laminated glass", "tempered"], ["GL", "GLZ", "IGU"]),
    "insulation": ("insulation", ["insulation", "insulated", "*dämmung", "*dämmstoff", "*dämmplatte", "*wärmedämm", "*isolier", "isolation", "isolant", "isolament",
                                  "*isolatie", "*isolering", "*eriste", "aislamiento", "izolac", "izolat", "mineral wool", "*mineralwolle",
                                  "rock wool", "rockwool", "glass wool", "glasswool", "*glaswolle", "*mineraalivilla",
                                  "*mineraluld", "eps", "xps", "polystyren", "polyurethan", "styrofoam", "kingspan",
                                  "thermal barrier", "*steinwolle", "*steenwol", "*steinull"], ["INSUL", "INS", "RIGID", "BATT"]),
    "tile": ("tile", ["tile", "tiles", "tiled", "tiling", "ceramic", "porcelain", "*fliese", "*fliesen", "*kachel", "carrelage",
                      "carreau", "faience", "piastrell", "*tegels", "*fliser", "*flis", "*laatta", "*laatoitus", "*kaakeli",
                      "*klinkkeri", "azulejo", "baldosa", "ceramica", "*plytk", "faianta", "gresie", "quarry tile", "mosaic"],
             ["CT", "QT", "CER", "ACT", "PORC"]),
    "parquet": ("parquet / wood flooring", ["parquet", "parkett", "parket", "*parketti", "*parkiet", "parchet", "hardwood floor",
                                            "wood floor", "wood flooring", "*holzboden", "*dielen", "plancher", "*lautalattia",
                                            "floorboard", "plank floor"], ["HWF"]),
    "laminate": ("laminate", ["laminate", "*laminat", "stratifie", "laminato", "*laminaatti", "laminado"], ["LAM"]),
    "resilient": ("vinyl / resilient flooring", ["vinyl", "linoleum", "lino", "resilient", "pvc floor", "*muovimatto", "*vinyyli",
                                                 "*kunststoffbelag", "lvt", "vct", "sheet vinyl", "rubber floor",
                                                 "*gummi", "caoutchouc", "*kumi", "marmoleum", "*linoleo", "*linoleum"],
                  ["VCT", "LVT", "SV", "RB", "RF", "RES", "LVP"]),
    "carpet": ("carpet", ["carpet", "*teppich", "moquette", "tapis", "*tapijt", "*taeppe", "*teppe", "*matto", "alfombra",
                          "moqueta", "wykladzin", "mocheta", "*kokolattiamatto"], ["CPT", "CP", "CARP"]),
    "screed": ("screed / concrete topping", ["screed", "*estrich", "chape", "massetto", "*dekvloer", "*avjamning", "*tasoite",
                                             "*pintavalu", "solera", "wylewk", "sapa", "*zementestrich", "*anhydrit"], []),
    "epoxy": ("epoxy / resin", ["epoxy", "*epoxid", "resin", "*harz", "resine", "resina", "hars", "*epoksi", "polyurea",
                                "*kunstharz", "sealed concrete", "stained and sealed"], ["EPX", "EP"]),
    "membrane": ("membrane / waterproofing", ["membrane", "epdm", "tpo", "bitumen", "*bitum", "asphalt", "waterproofing", "*abdichtung",
                                              "etancheite", "impermeabil", "roofing felt", "*dachbahn", "*dakbedekking", "*kermikate",
                                              "vapour barrier", "vapor barrier", "*dampfsperre", "*dampbarriere", "air barrier",
                                              "house wrap", "tyvek"], ["EPDM", "TPO", "VB", "AB"]),
    "plastic": ("plastic", ["plastic", "*kunststoff", "plastique", "plastica", "*kunststof", "*muovi", "polyethylene", "pvc",
                            "pe-x", "pex", "polypropylene", "acrylic", "polycarbonate", "frp", "fiberglass", "fibreglass", "mlcp",
                            "multi-layer composite"], ["PVC", "PE", "PP", "FRP", "PEX"]),
    "paint": ("paint", ["paint", "painted", "*farbe", "*anstrich", "peinture", "pittura", "verf", "*maling", "maali", "pintura",
                        "*farba", "vopsea", "lacquer", "*lackier", "*lakk", "coating"], ["PNT", "PTD", "PT"]),
    "earth": ("earth / soil", ["earth", "soil", "erde", "*erdreich", "terre", "terreno", "grond", "jord", "*maa-aines", "tierra",
                               "grass", "turf", "rasen", "gazon", "gras", "*nurmi", "planting", "green roof", "*begrünt",
                               "*dachbegrünung", "artificial turf"], []),
    "gravel": ("gravel", ["gravel", "kies", "gravier", "ghiaia", "grind", "grus", "sora", "grava", "*zwir", "pietris",
                          "crushed stone", "*schotter"], ["GRVL"]),
    "cork": ("cork", ["cork", "*korkboden", "*korkplatte", "liege", "sughero", "*korkki", "corcho"], []),
    "composite": ("composite", ["composite", "verbund", "wpc", "trex", "acp", "fibre cement", "fiber cement", "*faserzement",
                                "eternit", "hardie"], ["FC"]),
}

# a family that is a kind of another: "where is concrete used?" also finds concrete block, screed and stained concrete
PARENTS = {"block": ["concrete", "masonry"], "brick": ["masonry"], "stone": ["masonry"], "screed": ["concrete"],
           "parquet": ["wood"], "plywood": ["wood"], "aluminium": ["metal"], "steel": ["metal"], "laminate": [],
           "epoxy": [], "resilient": [], "carpet": [], "tile": []}
LABEL = {k: v[0] for k, v in FAMILIES.items()}

# what a question may call the finish of a surface, and the surface it means
FINISH_WORDS = {
    "floor": r"floor(?:ing)?|floor (?:finish|covering)|bodenbelag|fu(?:ss|ß)boden(?:belag)?|bodenbel\w*|revetement de sol|sol|"
             r"pavimento|pavimentazione|vloer(?:afwerking|bedekking)?|gulv(?:belaegning)?|golv|lattia(?:materiaali|pinnoite)?|"
             r"porand|podlog\w*|pardoseal\w*|suelo|piso",
    "wall": r"wall(?:s)? (?:finish|covering|lining)|wandbelag|wandbekleidung|revetement mural|rivestimento (?:della )?parete|"
            r"wandafwerking|veggbeklaedning|seinapinta|wall finish\w*|walls? finish\w*",
    "ceiling": r"ceiling(?:s)?|decke(?:nbekleidung)?|plafond|soffitto|controsoffitto|loft|katto|alakatto|lagi|sufit|tavan|techo|cielorraso",
    "base": r"base(?:board)?|skirting|sockel(?:leiste)?|plinthe|battiscopa|plint|sokkel|jalkalista|rodapie|zocalo",
}

MATERIAL_WORD = re.compile(
    r"\b(?:materials?|made (?:of|from|out of)|built (?:of|from|with)|construct(?:ed|ion) (?:of|from|with)|"
    r"material composition|composition of|composed of|finish(?:es|ed|ing)?|cladding|clad (?:in|with)|lining|covering|surfac(?:e|ing) material|"
    r"what (?:is|are) (?:the )?\w+ (?:made|built)|"
    r"materiali?|material(?:ien|es|e)?|baustoff\w*|werkstoff\w*|beschaffenheit|aus (?:was|welchem)|woraus|"
    r"mat[ée]riaux|mat[ée]riau|en quoi|fait (?:en|de)|di (?:che|cosa)|materiaal|materialen|materiale[rt]?|materjal\w*|"
    r"materia(?:ł|l)\w*|materiaali\w*|mist[aä] .* tehty|de qu[ée]|hecho de|din ce|"
    r"belag|bekleidung|oberfl[äa]che|rev[êe]tement|rivestimento|afwerking|bekleding|bel[æa]gning|pinnoite|pintamateriaali|"
    r"acabado|wyko(?:ń|n)czeni\w*|finisaj)\b", re.I)


def _stem_regex(stem: str) -> str:
    inner = stem.startswith("*")
    s = re.escape(fold(stem.lstrip("*"))).replace(r"\ ", r"[\s\-]*")
    return (r"" if inner else r"(?<![a-z0-9])") + s


_FAMILY_RX = {k: re.compile("|".join(_stem_regex(s) for s in v[1])) for k, v in FAMILIES.items()}
_ABBR = {a: k for k, v in FAMILIES.items() for a in v[2]}
_ABBR_RX = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted((re.escape(a) for a in _ABBR), key=len, reverse=True)) +
                      r")(?![a-z])\.?(?:-?\d+[A-Z]?)?(?![A-Za-z])")
# short words that are also abbreviations: only trusted as a tag ("WD-1") or inside a schedule's material column
WEAK_ABBR = {"AL", "GL", "ST", "SS", "CT", "PT", "CB", "MW", "KS", "BR", "GB", "GK", "RB", "RF", "VB", "SV", "AB", "FC", "EP",
             "PE", "PP", "CU", "ZN", "RC", "HM", "CP", "INS", "MAS", "MET", "RES", "LAM", "BET", "HW", "ALU"}


@dataclass
class Hit:
    family: str
    text: str           # the matched words as printed
    start: int = 0
    end: int = 0
    abbreviation: bool = False


def families_in(text: str, abbreviations: bool = False, strict: bool = True) -> list[Hit]:
    """Material families named in a piece of text. Abbreviations (CONC, GWB, WD-1) only when asked: in sheet text they are
    common, in questions they are rare and short ones collide with ordinary words. `strict`: short abbreviations (AL,
    GL, CT...) need a tag number or a schedule column to count."""
    if not text:
        return []
    f = fold(text)
    hits: list[Hit] = []
    for fam, rx in _FAMILY_RX.items():
        for m in rx.finditer(f):
            hits.append(Hit(fam, text[m.start():m.end()] if len(f) == len(text) else m.group(0), m.start(), m.end()))
    if abbreviations:
        for m in _ABBR_RX.finditer(text):
            tok = m.group(1)
            tagged = m.group(0) != tok and re.search(r"\d", m.group(0)) is not None
            if strict and tok in WEAK_ABBR and not tagged:
                continue
            if not tok.isupper():
                continue
            hits.append(Hit(_ABBR[tok], m.group(0), m.start(), m.end(), True))
    # keep the most specific family where two overlap ("concrete block" is block, not concrete; "glass wool" insulation)
    hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
    out: list[Hit] = []
    for h in hits:
        if any(o.start <= h.start and h.end <= o.end and (o.end - o.start) > (h.end - h.start) for o in out):
            continue
        if any(o.start == h.start and o.end == h.end and o.family == h.family for o in out):
            continue
        out.append(h)
    return out


def family_set(text: str, abbreviations: bool = False, strict: bool = True) -> set[str]:
    return {h.family for h in families_in(text, abbreviations, strict)}


def matches(families: set[str], wanted: set[str]) -> bool:
    """A thing made of `families` counts as `wanted` (a family or its parents: block counts as concrete)."""
    if not wanted:
        return bool(families)
    expanded = set(families)
    for f in families:
        expanded.update(PARENTS.get(f, []))
    return bool(expanded & wanted)


# ------------------------------------------------------------------ questions

ELEMENT_WORDS = {
    "wall": r"walls?|wand|w(?:ä|ae)nde|wandtyp\w*|murs?|mur(?:o|i)|muur|muren|v(?:æ|ae)gg?e?|vegg(?:er)?|v(?:ä|a)gg(?:ar)?|sein(?:a|ä|ad)\w*|"
            r"[sś]cian\w*|pere(?:te|ti)|pared(?:es)?|partitions?",
    "slab": r"slabs?|decken(?:platte)?n?|dalles?|solai?o|solette?|vloerplat\w*|d(?:æ|ae)k|dekke|laatat?|plate|losas?|p(?:ł|l)yt\w*|placa",
    "floor": r"floors?|flooring|b(?:ö|oe)den|fu(?:ss|ß)b(?:ö|oe)den|planchers?|pavimenti?|vloer(?:en)?|gulv|golv|lattia\w*|p(?:õ|o)rand|"
             r"pod(?:ł|l)og\w*|pardoseal\w*|suelos?",
    "ceiling": r"ceilings?|decke(?:n)?(?:bekleidung)?|plafonds?|soffitt[io]|controsoffitt[io]|lofter?|katto|kato(?:t|issa)|lagi|sufit\w*|tavan|techos?",
    "roof": r"roofs?|roofing|d(?:a|ä)ch(?:er)?|toits?|toiture|tett[oi]|copertura|dak(?:en)?|tag(?:et)?|tak(?:et)?|katus|vesikatto|dach\w*|"
            r"acoperi\w*|tejados?|cubierta",
    "door": r"doors?|t(?:ü|ue)r(?:en)?|portes?|port[ae]|deur(?:en)?|d(?:ø|o)r(?:e|er)?|uks\w*|drzwi|u(?:ș|s)(?:a|i)|puertas?|ov(?:i|et)",
    "window": r"windows?|fenster|fen(?:ê|e)tres?|finestr[ae]|ramen|raam|vinduer?|f(?:ö|o)nster|aken\w*|okn\w*|ferestr\w*|ventanas?|ikkun\w*",
    "column": r"columns?|st(?:ü|ue)tzen?|s(?:ä|ae)ulen?|poteaux?|pilastr[oi]|colonn[ae]|kolom(?:men)?|s(?:ø|o)jler?|sammas|s(?:ł|l)up\w*|"
              r"st(?:â|a)lp\w*|columnas?|pilar(?:i|it)?",
    "beam": r"beams?|tr(?:ä|ae)ger|balken|poutres?|trav[ei]|balk(?:en)?|bj(?:æ|ae)lker?|tala|belk\w*|grinz\w*|grind(?:a|ă)|vigas?|palk\w*|girders?|joists?",
    "stair": r"stairs?|staircases?|treppen?|escaliers?|scal[ae]|trap(?:pen)?|trapper?|trepp\w*|schod\w*|scar(?:a|i)|escaleras?|portaat|portaikko",
    "railing": r"railings?|handrails?|gel(?:ä|ae)nder|garde-corps|ringhier\w*|balustr\w*|leuning\w*|r(?:æ|ae)kv(?:æ|ae)rk|kaide\w*|barandill\w*",
    "foundation": r"foundations?|footings?|fundament\w*|fondations?|fondazion[ei]|funderingen?|perustu\w*|cimentaci\w*|piles?",
    "covering": r"coverings?|finishes|bekleidung\w*|rev(?:ê|e)tements?|rivestiment[oi]|bekleding\w*|pinnoit\w*",
    "facade": r"fa(?:ç|c)ades?|fassaden?|gevel\w*|julkisivu\w*|cladding|elevations?",
    "pipe": r"pipes?|piping|pipework|rohre?|rohrleitung\w*|tuyaux?|tubazion\w*|leiding\w*|r(?:ø|o)r|putk\w*|tuber(?:í|i)as?",
    "duct": r"ducts?|ductwork|kan(?:ä|ae)le?|luftkan\w*|gaines?|condott\w*|kanal\w*|ilmakanav\w*|conductos?",
    "conduit": r"conduits?|cable trays?|kabel\w*|cables?|cablage|cavi|kaapel\w*",
    # "structural system", "load-bearing": not "insulation in their structure"
    "structure": r"structural|load[- ]?bearing|structural system|primary structure|tragwerk\w*|tragend\w*|ossature|strutturale|"
                 r"draagconstructie|kantav\w*|kantava rakenne|estructural",
    "insulation_el": r"insulation layers?",
}
EXTERIOR = re.compile(r"\b(?:exterior|external|outer|outside|facade|fa[çc]ade|envelope|au(?:ss|ß)en\w*|ext(?:é|e)rieur\w*|estern[oi]|"
                      r"buiten\w*|ydre|ytter\w*|v(?:ä|a)lis\w*|zewn(?:ę|e)trzn\w*|ulko\w*|exteriores?)\b", re.I)
INTERIOR = re.compile(r"\b(?:interior|internal|inner|inside|innen\w*|int(?:é|e)rieur\w*|intern[oi]|binnen\w*|indre|inner\w*|sise\w*|"
                      r"wewn(?:ę|e)trzn\w*|sis(?:ä|a)\w*|interiores?|partition\w*)\b", re.I)
BEARING = re.compile(r"\b(?:load[- ]?bearing|structural|bearing|tragend\w*|porteurs?|portant[ei]|dragend\w*|b(?:æ|ae)rende|kantav\w*|"
                     r"no(?:ś|s)n\w*|portante)\b", re.I)
NON_BEARING = re.compile(r"\b(?:non[- ]?(?:load[- ]?)?bearing|non[- ]?structural|nichttragend\w*|nicht tragend\w*|non porteurs?|"
                         r"niet[- ]?dragend\w*|non portant[ei])\b", re.I)
WHERE = re.compile(r"\b(?:where|which (?:rooms?|spaces?|areas?|elements?|parts?|walls?|floors?|levels?|storeys?|places?)|in which|"
                   r"wo\b|in welche\w*|welche (?:r(?:ä|ae)ume|bauteile)|o(?:ù|u)\b|dans quel\w*|dove|in quali|waar|hvor|var\b|kus\b|"
                   r"gdzie|unde|d(?:ó|o)nde|miss(?:ä|a)|mihin|used (?:in|on|for)|applied (?:in|to|on)|locations?)\b", re.I)
COUNT = re.compile(r"(?:^\s*|[,;:]\s+)(?:how many|wie viele|wieviele|combien|quant[ie]|hoeveel|hvor mange|hur m(?:å|a)nga|ile|c(?:â|a)te|kui mitu|"
                   r"cu(?:á|a)nt[oa]s|kuinka monta|montako|number of|count)\b", re.I)
QUANTITY = {
    "volume": re.compile(r"\b(?:volume|volumen|volumes|rauminhalt|kubatur|tilavuus|maht|obj(?:ę|e)to(?:ść|sc)|m3|m³|cu\.? ?m|cubic)\b", re.I),
    "area": re.compile(r"\b(?:area|areas|fl(?:ä|ae)che|surface|superficie|oppervlakte|areal|pinta-ala|pindala|powierzchni\w*|suprafa\w*|"
                       r"m2|m²|sq\.? ?m|sqm|square (?:met\w*|feet|foot)|sf)\b", re.I),
    "length": re.compile(r"\b(?:length|l(?:ä|ae)nge|longueur|lunghezza|lengte|l(?:æ|ae)ngde|pituus|pikkus|d(?:ł|l)ugo(?:ść|sc)|lungime|"
                         r"longitud|metres of|meters of|running)\b", re.I),
    "thickness": re.compile(r"\b(?:thick(?:ness)?|dicke|st(?:ä|ae)rke|(?:é|e)paisseur|spessore|dikte|tykkelse|tjocklek|paksuus|"
                            r"paksus|grubo(?:ść|sc)|grosime|espesor)\b", re.I),
}


# questions that mention a material but ask for one of its properties: the planner reads those from property sets
NOT_MATERIAL = re.compile(r"\b(?:lifespan|life span|expected life|service life|durability|cost|price|budget|carbon|co2|embodied|"
                          r"recycl\w*|reus\w*|end[- ]of[- ]life|fire (?:rating|resistance)|u-?value|r-?value|thermal transmittance|"
                          r"acoustic rating|manufacturer|supplier|warranty|density|weight|mass|strength|grade|class)\b", re.I)


AMOUNT = re.compile(r"\b(?:total amount|amount of|how much|quantity of|quantities of|wie viel|combien de|quanto|"
                    r"hvor meget|kuinka paljon|cu(?:á|a)nto)\b", re.I)


@dataclass
class Query:
    ask: str                                   # list | where | count | quantity | thickness
    families: set[str] = field(default_factory=set)
    elements: list[str] = field(default_factory=list)
    side: str | None = None                    # exterior | interior
    bearing: bool | None = None
    quantity: str | None = None                # volume | area | length
    surface: str | None = None                 # floor | wall | ceiling | base: a finish question
    text: str = ""

    def describe(self) -> str:
        what = " / ".join(LABEL[f] for f in sorted(self.families)) or "materials"
        el = " and ".join(self.elements) or "the drawing"
        side = f"{self.side} " if self.side else ""
        return f"{self.ask}: {what} in {side}{el}"


# German / Dutch / Nordic compounds name the element at the end of a word: Außen*wände*, Zwischen*decke*, Flach*dach*
COMPOUND = {"wall": r"w(?:a|ae)nde?n?|wanden|vegger?|vaegge?|seinat?", "ceiling": r"decken?|plafonds?|katot?",
            "roof": r"d(?:a|ae)ch(?:er)?|daken|taget|katto", "floor": r"b(?:o|oe)den|vloeren?|gulve?t?|lattiat?",
            "door": r"tur(?:en)?|tueren|deuren?", "window": r"fenster|ramen|vinduer?|ikkunat?",
            "column": r"st(?:u|ue)tzen?|kolommen?|pilarit?", "beam": r"tr(?:a|ae)ger|balken|bjaelker?|palkit?",
            "stair": r"treppen?|trappen?|portaat", "slab": r"platten?|platen|laatat"}


def elements_in(question: str) -> list[str]:
    f = fold(question)
    out = []
    for k, pat in ELEMENT_WORDS.items():
        if re.search(r"(?<![a-z])(?:" + fold(pat) + r")(?![a-z])", f) or (
                k in COMPOUND and re.search(r"[a-z]{3,}(?:" + COMPOUND[k] + r")(?![a-z])", f)):
            out.append(k)
    if "floor" in out and re.search(r"\b(?:floor (?:plan|area)|ground floor|first floor|second floor|top floor|upper floor|"
                                     r"each floor|per floor|on (?:the|this|which) floor|floors? \d|level)\b", question, re.I):
        # "floor" as a storey, not a floor surface
        if not re.search(r"floor (?:finish|covering|material)|flooring", question, re.I):
            out.remove("floor")
    return out


def intent_of(question: str) -> Query | None:
    """A material question, parsed without an LLM; None when the question is not about materials."""
    q = question.strip()
    # material words inside quoted names ('SOIL. LIN. (1E30)', '00 Begane grond') or asides ("(including glazed
    # balconies)") describe what is counted, not what it is made of; an aside next to a schedule code still counts
    # ("door type B1 (aluminium door with glazing)")
    core = re.sub(r"['\"‘’“”«»„][^'\"‘’“”«»„]{1,80}['\"‘’“”«»„]", " ", q)
    asides = re.findall(r"\(([^)]*)\)", core)
    core = re.sub(r"\([^)]*\)", " ", core)
    fams = family_set(core, abbreviations=True, strict=True)
    if not fams and asides and re.search(r"\b[A-Z]{1,3}-?\d{1,2}[A-Z]?\b", core):
        fams = family_set(" ".join(asides), abbreviations=True, strict=True)
    # a sheet's name is not a material question: "how many rooms are in the finish room schedule?"
    bare = re.sub(r"\b(?:room )?finish(?:es)? (?:room )?schedules?\b|\bdoor (?:and (?:room )?finish )?schedules?\b|"
                  r"\bfinish(?:ed)? floor (?:level|elevation)\b|\bf\.?f\.?l\.?\b", " ", q, flags=re.I)
    worded = MATERIAL_WORD.search(bare) is not None
    surface = None
    for s, pat in FINISH_WORDS.items():
        if re.search(r"(?<![a-z])(?:" + fold(pat) + r")(?![a-z])", fold(bare)) and re.search(
                r"finish|covering|belag|bekleid|revetement|rivestiment|afwerking|bekleding|pinnoit|material|flooring|surface|"
                r"acabado|wykonczen|made of|what kind|what type|type of", fold(bare)):
            surface = s
            break
    if not fams and not worded and surface is None:
        return None
    if NOT_MATERIAL.search(q) and not COUNT.search(q):
        return None    # lifespan, cost, carbon, recyclability, fire rating of a material: properties, not the material
    worded = worded or surface is not None
    els = elements_in(q)
    side = "exterior" if EXTERIOR.search(q) and not INTERIOR.search(q) else ("interior" if INTERIOR.search(q) and not EXTERIOR.search(q) else None)
    bearing = False if NON_BEARING.search(q) else (True if BEARING.search(q) else None)
    if "structure" in els and bearing is None:
        bearing = True
    quantity = next((k for k, rx in QUANTITY.items() if rx.search(q)), None)
    if quantity is None and AMOUNT.search(q) and (fams or surface):
        # "total amount of tiling", "how much concrete": the natural measure of what holds it
        quantity = ("area" if (surface or set(els) & {"floor", "ceiling", "covering", "wall", "facade", "roof"})
                    and not set(els) & {"slab", "foundation", "column", "beam"} else
                    "length" if set(els) & {"pipe", "duct", "conduit"} else "volume")
    if quantity == "thickness":
        ask = "thickness" if fams else "list"
        quantity = None
    elif quantity and fams:
        ask = "quantity"
    elif COUNT.search(q) and fams:
        ask = "count"
    elif WHERE.search(q) and fams and not worded:
        ask = "where"
    elif fams and not worded and not els:
        ask = "where"          # "concrete?" / "Is there any carpet?": where it is
    elif fams and not worded and re.match(r"^\s*(?:is|are|does|do|has|have|gibt es|y a-t-il|ci sono|er|finns|onko|czy)\b", q, re.I):
        ask = "count"          # "are there any steel beams?": a yes/no answered by a count
    else:
        ask = "list"
    if quantity and not fams:
        ask = "quantity"       # "total volume of insulation material" has a family; "volume of each material" lists
    return Query(ask, fams, els, side, bearing, quantity, surface, q)


# ------------------------------------------------------------------ answers

@dataclass
class Finding:
    """One place a material shows on the drawing."""
    family: str                    # FAMILIES key, or "" when the material named has no family
    material: str                  # as the source names it: "Masonry - Brick", "F4 - RESILIENT FLOORING", "concrete hatch"
    where: str                     # what / where: "exterior wall type 'Brick on CMU'", "room 113 AEROBICS & YOGA (floor)"
    basis: str = "direct"          # direct | inferred | review
    source: str = ""               # ifc | type-name | schedule | note | legend | cad-hatch | hatch | vision
    evidence: list[Evidence] = field(default_factory=list)
    quantity: float | None = None
    unit: str | None = None


def summarise(findings: list[Finding], limit: int = 14) -> str:
    """Findings -> 'Concrete: room 101 floor (F1 stained and sealed concrete), ...; Carpet: ...'."""
    by: dict[str, list[Finding]] = {}
    for f in findings:
        by.setdefault(f.material, []).append(f)
    parts = []
    for mat, fs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        wheres = []
        for f in fs:
            if f.where and f.where not in wheres:
                wheres.append(f.where)
        shown = ", ".join(wheres[:6]) + (f" and {len(wheres) - 6} more" if len(wheres) > 6 else "")
        parts.append(f"{mat}: {shown}" if shown else mat)
        if len(parts) >= limit:
            parts.append(f"... {len(by) - limit} more materials")
            break
    return "\n".join(parts)


def answer(d, question: str, sheet: tuple[str, int] | None = None, use_llm: bool = True) -> tuple[Answer | None, list[dict]]:
    """Answer a material question about a drawing. Returns (answer or None when this is not a material question,
    the specialists asked: [{source, outcome, note}]). A None value inside the Answer = searched everything, found nothing."""
    q = intent_of(question)
    if q is None:
        return None, []
    from . import materials_ifc, materials_sheet

    tried: list[dict] = []

    def note(source: str, a: Answer | None) -> None:
        tried.append({"source": source, "outcome": "answered" if a is not None and a.value is not None else "nothing",
                      "note": (a.method if a is not None else "")[:200]})

    model = materials_ifc.has_model(d)
    # 1. the BIM model behind the drawing (IFC drawings, storey plans rendered from IFC)
    if model == "own":
        a = materials_ifc.answer(d, q)
        note("model", a)
        if a is not None and a.value is not None:
            return a, tried
    # 2. the sheet's printed text (schedules, notes, labels, legends), incl. the other sheets of its set
    a = materials_sheet.answer(d, q, sheet)
    note("sheet text", a)
    if a is not None and a.value is not None:
        return a, tried
    # 3. hatches: CAD hatch names, or textures on the sheet image (trained classifier)
    from . import hatch

    a = hatch.answer(d, q, sheet)
    note("hatches", a)
    if a is not None and a.value is not None:
        return a, tried
    # 3b. a drawing set's building model (opening_design): after the sheet, since the sheet is what was asked about;
    # not for one named door / room the sheet set could not find (a list of every door in the model is no answer)
    specific = re.search(r"\b(?:door|room|type|mark)\s+\(?[A-Z]?\d", q.text, re.I) is not None
    if model == "project" and not specific:
        a = materials_ifc.answer(d, q)
        note("project model", a)
        if a is not None and a.value is not None:
            return a, tried
    # 4. the vision reader, with marks
    if use_llm:
        from . import materials_vision

        a = materials_vision.answer(d, q, sheet)
        note("vision", a)
        if a is not None and a.value is not None:
            return a, tried
    searched = "; ".join(f"{t['source']}: {t['note'] or 'nothing'}" for t in tried)
    return Answer(None, None, "no_material", 0.0,
                  f"No material is named or drawn for this on the drawing. Searched — {searched}"), tried
