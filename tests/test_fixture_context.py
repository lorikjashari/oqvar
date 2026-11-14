"""Room evidence is backed by distinct nearby fixtures."""
from stoa_qa import cubicasa, fixture_context as fc

THRESHOLDS = dict.fromkeys(fc.FIXTURES, 0.25)


def box(name, coords, score=0.8):
    return [fc.IDS[name], score, *coords]


def test_room_needs_multiple_fixture_types_and_uses_each_fixture_once():
    room = box("bathroom", [0, 0, 90, 90])
    nearby = box("bathroom", [75, 0, 150, 90])
    toilet = box("toilet", [20, 20, 30, 35])
    basin = box("sink", [35, 20, 45, 30])
    screen = box("shower_screen", [50, 45, 70, 65])
    shower = box("shower", [55, 50, 75, 70], 0.93)
    assert fc.room_context([room, nearby, toilet, basin, screen], THRESHOLDS, 0.3) == []
    matches = fc.room_context([room, nearby, toilet, basin, screen, shower], THRESHOLDS, 0.3)
    assert len(matches) == 1
    assert matches[0].fixture_types == {"toilet", "sink", "shower"}


def test_washing_machine_wins_against_a_false_toilet_at_the_same_position():
    toilet = box("toilet", [10, 10, 25, 28], 0.56)
    machine = box("washing_machine", [10, 10, 26, 29], 0.82)
    separate = box("toilet", [40, 40, 50, 53], 0.72)
    assert fc.selected_fixtures([toilet, machine, separate], THRESHOLDS) == [separate]


def test_close_fixture_group_creates_one_candidate_without_a_room_box():
    toilet = box("toilet", [20, 20, 30, 35])
    basin = box("sink", [35, 20, 45, 30])
    shower = box("shower", [55, 30, 75, 50])
    kitchen = box("sink", [150, 100, 170, 120])
    matches = fc.fixture_groups([toilet, basin, shower, kitchen], [])
    assert len(matches) == 1 and matches[0].fixtures == [toilet, basin, shower]
    assert fc.fixture_groups([toilet, basin, shower, kitchen], matches) == []


def test_fixture_questions_keep_distinct_classes():
    assert cubicasa.upload_question_key("Highlight all wash basins") == "sinks"
    assert cubicasa.upload_question_key("How many showers?") == "showers"
    assert cubicasa.upload_question_key("How many bathtubs?") == "bathtubs"
    assert cubicasa.upload_question_key("How many showers and bathtubs?") == "baths"
