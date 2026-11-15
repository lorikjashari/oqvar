"""Spot-check stoa_qa.materials question parsing and vocabulary on hand-picked questions and strings."""
from stoa_qa import materials as M

QUESTIONS = [
    "Which materials are used to construct the walls and their usage across wall types?",
    "What materials are used on the building's exterior walls?",
    "What is the total volume of concrete used in floor slabs?",
    "What is the thickness of the insulation of the external walls?",
    "What is the wall finishing material in treatment room 1D34?",
    "Where is concrete used?",
    "How many walls are brick?",
    "Which ceiling materials are used in the building?",
    "What type of insulation is used in the external walls?",
    "How many rooms are there?",
    "What is the area of the kitchen?",
    "Welche Materialien haben die Außenwände?",
    "Quel est le revêtement de sol de la cuisine?",
    "Mistä materiaalista lattia on tehty?",
    "Is there any carpet in the offices?",
    "How many doors in the door schedule have a wood solid core door (material WD-1)?",
    "What is the installation height without the window?",
    "What is the expected lifespan of the roofing material?",
    "Are there any steel beams spanning the living room of Duplex B?",
    "What kind of flooring is in the kitchen?",
    "Which rooms have tile floors?",
    "How many windows are on the ground floor?",
    "What is the net floor area of the apartment?",
    "Which room is adjacent to the kitchen?",
]
STRINGS = ["Kalksandstein 2774059904 300", "Basiswand:STB 250", "Trockenbau - Rigips Die Blaue RFI", "PAINTED GYP. BD",
           "WD-1 | STL-1", "AL-2/GL-2", "Leichtbeton - Mauerwerk", "DD Gipsplaat", "Masonry - Concrete Block",
           "installation without werden Korkeus Zeichen Kasten heating episode", "F4 | RESILIENT FLOORING (T.B.D.)",
           "STAINED AND SEALED CONCRETE FLOOR", "Isolierung, hart 275646950", "Betong bærende", "DD gasbeton",
           "Dachdeckung - Kies", "Titanzink - Eindeckung", "PARKETTI", "KLINKKERI", "Ortbeton - bewehrt Verputzt"]

for t in QUESTIONS:
    q = M.intent_of(t)
    print(t[:72].ljust(72), (q.describe() + f" | qty={q.quantity} surface={q.surface} side={q.side}") if q else None)
print()
for t in STRINGS:
    print(t.ljust(48), [(h.family, h.text) for h in M.families_in(t, abbreviations=True)])
