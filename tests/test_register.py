"""How the turn is pitched.

The thing under test is mostly restraint, so most of these assert silence. A
register note on every turn would be a tic, and the model would start
performing attentiveness rather than paying it.
"""

import time

from vesper.register import (
    LONG_SESSION_S,
    MID_FLOW_GAP_S,
    Register,
    attach,
)

def clock(hour: int, minute: int = 0) -> time.struct_time:
    return time.struct_time((2026, 9, 3, hour, minute, 0, 2, 246, -1))

# --- silence is the default -------------------------------------------------

def test_an_ordinary_turn_says_nothing():
    r = Register(started_at=0.0)
    r.note_turn("Code.exe", now=10.0)
    assert r.line(now=20.0, clock=clock(14)) == ""

def test_a_first_turn_in_the_afternoon_says_nothing():
    assert Register(started_at=0.0).line(now=5.0, clock=clock(11)) == ""

# --- the signals ------------------------------------------------------------

def test_late_and_long_asks_for_less():
    r = Register(started_at=0.0)
    line = r.line(now=LONG_SESSION_S + 1, clock=clock(23, 40))
    assert "shorter than usual" in line

def test_late_on_its_own_still_asks_for_short():
    r = Register(started_at=0.0)
    assert "keep it short" in r.line(now=60.0, clock=clock(1, 15))

def test_the_small_hours_count_as_late():
    """00:30 is later than 23:30, and a naive hour check gets that backwards."""
    r = Register(started_at=0.0)
    assert r.line(now=60.0, clock=clock(0, 30)) != ""

def test_mid_exchange_drops_the_preamble():
    r = Register(started_at=0.0)
    r.note_turn("Code.exe", now=100.0)
    r.note_turn("Code.exe", now=110.0)
    assert "no preamble" in r.line(now=120.0, clock=clock(14))

def test_a_long_gap_is_not_mid_exchange():
    r = Register(started_at=0.0)
    r.note_turn("Code.exe", now=100.0)
    r.note_turn("Code.exe", now=110.0)
    assert "no preamble" not in r.line(now=110.0 + MID_FLOW_GAP_S + 1, clock=clock(14))

def test_coming_back_from_away_asks_it_not_to_recap():
    r = Register(started_at=0.0)
    r.note_idle(1200.0)
    assert "do not recap" in r.line(now=10.0, clock=clock(14))

def test_being_away_is_true_for_exactly_one_turn():
    """Otherwise every turn for the rest of the session says they just got back."""
    r = Register(started_at=0.0)
    r.note_idle(1200.0)
    assert "do not recap" in r.line(now=10.0, clock=clock(14))
    assert "do not recap" not in r.line(now=11.0, clock=clock(14))

def test_staying_on_one_thing_asks_it_to_stay_out_of_the_way():
    r = Register(started_at=0.0)
    for tick in range(4):
        r.note_turn("Code.exe", now=float(tick))
    assert "stay out of the way" in r.line(now=500.0, clock=clock(14))

def test_switching_windows_resets_that():
    r = Register(started_at=0.0)
    for tick in range(4):
        r.note_turn("Code.exe", now=float(tick))
    r.note_turn("chrome.exe", now=5.0)
    assert "stay out of the way" not in r.line(now=500.0, clock=clock(14))

# --- failure changes the subject --------------------------------------------

def test_repeated_failure_crowds_everything_else_out():
    """When something is not working, how long the session has run stops mattering."""
    r = Register(started_at=0.0)
    r.note_failure()
    r.note_failure()
    line = r.line(now=LONG_SESSION_S + 1, clock=clock(23, 50))
    assert "did not work" in line
    assert "Change approach" in line
    assert "shorter than usual" not in line, "the failure is the whole message"

def test_one_failure_is_a_coincidence_not_a_pattern():
    r = Register(started_at=0.0)
    r.note_failure()
    assert "did not work" not in r.line(now=10.0, clock=clock(14))

def test_success_clears_the_run():
    r = Register(started_at=0.0)
    r.note_failure()
    r.note_failure()
    r.note_success()
    assert r.line(now=10.0, clock=clock(14)) == ""

def test_it_never_tells_the_model_how_the_user_feels():
    """The whole design rests on this. Stated emotion is what makes it repellent.

    Every line is an instruction about delivery. None of them describe a mood,
    on either side, because "that sounds frustrating" is worse than silence.
    """
    banned = (
        "frustrat", "annoy", "stress", "tired", "upset", "sorry", "feel",
        "empathy", "apolog", "excited", "happy", "sad", "patient",
    )
    lines: list[str] = []
    r = Register(started_at=0.0)
    r.note_failure(); r.note_failure()
    lines.append(r.line(now=10.0, clock=clock(14)))
    r.note_success()
    r.note_idle(1200.0)
    lines.append(r.line(now=LONG_SESSION_S + 1, clock=clock(23, 50)))
    r2 = Register(started_at=0.0)
    r2.note_turn("Code.exe", now=1.0); r2.note_turn("Code.exe", now=2.0)
    lines.append(r2.line(now=3.0, clock=clock(2)))
    for line in lines:
        for word in banned:
            assert word not in line.lower(), f"{word!r} in {line!r}"

# --- cost -------------------------------------------------------------------

def test_the_line_stays_small_enough_to_ride_every_turn():
    """It is spoken into the same budget the sensor block is held to."""
    worst = Register(started_at=0.0)
    worst.note_idle(1200.0)
    for tick in range(6):
        worst.note_turn("Code.exe", now=float(tick))
    line = worst.line(now=LONG_SESSION_S + 1, clock=clock(23, 55))
    assert len(line) < 200, f"{len(line)} chars: {line!r}"

# --- attaching --------------------------------------------------------------

def test_attach_adds_nothing_when_there_is_nothing_to_say():
    assert attach("cpu 4%", "") == "cpu 4%"
    assert attach("cpu 4%", "   ") == "cpu 4%"

def test_attach_keeps_readings_and_guidance_apart():
    """Blurring them invites the model to read the cpu load as an instruction."""
    out = attach("cpu 4%", "keep it short")
    assert out == "cpu 4%\nhow to pitch this: keep it short"

def test_attach_works_with_no_context_at_all():
    assert attach("", "keep it short") == "how to pitch this: keep it short"

def test_the_combined_block_stays_inside_the_same_budget():
    """The sensor guard covers only its own half.

    `tests/test_privacy.py` holds `context_block` under 400 chars because it
    rides on every turn. The register rides on the same turns and was not
    covered by anything, so the two could drift into each other and quietly
    double what a turn costs.
    """
    from vesper.sensors import snapshot as sensors

    worst = Register(started_at=0.0)
    worst.note_idle(1200.0)
    for tick in range(6):
        worst.note_turn("Code.exe", now=float(tick))

    # A fixed window title rather than this machine's. `context_block()` reads
    # the live foreground window, so the budget being measured here included
    # whatever happened to be on screen at the time. It failed once against a
    # Chrome tab showing a long commit subject: 427 characters, none of them
    # the register's doing. What this test is about is the register's own
    # contribution, so the rest of the block is held still.
    ambient = "\n".join(
        "focused window: chrome.exe: a window title of ordinary length"
        if line.startswith("focused window:")
        else line
        for line in sensors.context_block().splitlines()
    )
    combined = attach(
        ambient,
        worst.line(now=LONG_SESSION_S + 1, clock=clock(23, 55)),
    )
    assert len(combined) < 400, f"{len(combined)} chars:\n{combined}"

def test_the_worst_case_is_the_failure_line():
    """Worth knowing which branch is the expensive one, so it stays the shortest."""
    stuck = Register(started_at=0.0)
    stuck.note_failure()
    stuck.note_failure()
    stuck.note_failure()
    assert len(stuck.line(now=10.0, clock=clock(23, 55))) < 200
