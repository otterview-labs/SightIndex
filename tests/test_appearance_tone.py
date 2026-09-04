"""Clothing tone and body shape from pixels: what they claim, and what they refuse to claim.

The extractor samples at pose keypoints, so these drive `_name` directly and stub the sampler
for the pipeline tests -- a synthetic rectangle has no shoulders for YOLO to find.
"""
import numpy as np
import pytest

from app.services.appearance_attributes import (
    AppearanceAttributeService,
    BodyShape,
    ClothingTone,
)


@pytest.fixture
def service():
    return AppearanceAttributeService(
        saturation_floor=110.0, hue_value_floor=90.0, dark_ratio=0.90
    )


def hsv(hue, saturation, value):
    return np.array([hue, saturation, value], dtype="float64")


def test_a_strong_colour_in_good_light_is_named(service):
    assert service._name(hsv(120, 200, 180), reference=120.0) == "blue"
    assert service._name(hsv(5, 200, 180), reference=120.0) == "red"


def test_a_black_shirt_is_dark_not_blue(service):
    """Measured on the live feed at saturation 136: near-black hue is noise, and it read blue."""

    assert service._name(hsv(115, 136, 40), reference=120.0) == "dark"


def test_a_strong_colour_in_shadow_is_not_named(service):
    """Every high-saturation patch in this footage came from shadow, where hue means nothing."""

    assert service._name(hsv(140, 150, 70), reference=120.0) == "dark"


def test_washed_out_clothing_gets_brightness_not_a_colour(service):
    assert service._name(hsv(133, 15, 160), reference=120.0) == "light"
    assert service._name(hsv(114, 50, 70), reference=120.0) == "dark"


def test_grey_is_never_claimed(service):
    """Grey is a statement about colour; low saturation means there is no colour to state."""

    assert service._name(hsv(0, 3, 128), reference=120.0) in {"light", "dark"}


def test_brightness_is_judged_against_the_crop_own_exposure(service):
    """The same shirt, shot in a bright corridor and a shaded doorway, is the same shirt."""

    shirt = hsv(120, 20, 100)

    assert service._name(shirt, reference=90.0) == "light"
    assert service._name(shirt, reference=160.0) == "dark"


def test_no_pose_means_no_claim(service, tmp_path, monkeypatch):
    """Without keypoints there is no way to know which pixels are clothing rather than wall."""

    import cv2

    path = tmp_path / "blank.png"
    assert cv2.imwrite(str(path), np.full((400, 100, 3), 128, np.uint8))
    monkeypatch.setattr(
        AppearanceAttributeService,
        "_keypoint_samples",
        lambda self, cv2_module, image, pose=None: None,
    )

    assert service.tone(path) is None
    assert service.describe(path) is None


def test_the_output_slots_into_the_existing_attribute_shape(service, tmp_path, monkeypatch):
    import cv2

    path = tmp_path / "body.png"
    assert cv2.imwrite(str(path), np.full((400, 100, 3), 128, np.uint8))
    monkeypatch.setattr(
        AppearanceAttributeService,
        "_keypoint_samples",
        lambda self, cv2_module, image, pose=None: (hsv(120, 200, 200), hsv(0, 5, 40)),
    )

    attributes = service.describe(path)

    assert attributes["object_type"] == "person"
    assert attributes["clothing"] == {"upper_color": "blue", "lower_color": "dark"}
    # The marker is what lets a later VLM pass know it may replace this.
    assert attributes["source"] == "cv_tone"


def test_an_unreadable_file_returns_nothing_rather_than_a_guess(service, tmp_path):
    broken = tmp_path / "not-an-image.jpg"
    broken.write_bytes(b"nope")

    assert service.tone(broken) is None
    assert service.describe(broken) is None


def test_light_and_dark_map_to_chinese_labels():
    from app.services.observation_index import attribute_labels

    labels = attribute_labels(
        {"object_type": "person", "clothing": {"upper_color": "light", "lower_color": "dark"}},
        "zh",
    )

    assert labels["上衣颜色"] == "浅色"
    assert labels["下装颜色"] == "深色"


@pytest.mark.parametrize(
    ("query", "field", "expected"),
    [
        ("浅色上衣", "upper_color", "light"),
        ("深色上衣", "upper_color", "dark"),
        ("深色下装", "lower_color", "dark"),
        ("浅色裤子", "lower_color", "light"),
        ("蓝色裤子", "lower_color", "blue"),
    ],
)
def test_the_query_parser_understands_what_the_reader_writes(query, field, expected, tmp_path):
    """Writing "light" into the index is pointless if no query can ask for it."""

    from app.config.settings import Settings
    from app.services.search import StructuredSearchService

    conditions = StructuredSearchService(None, Settings(data_dir=tmp_path)).parse_query(query)

    assert {c.field: c.values for c in conditions}[field] == (expected,)


def test_backfill_leaves_a_vlm_description_alone(monkeypatch, tmp_path):
    """A VLM knows gender, bags and behaviour; brightness must not replace that."""

    from fastapi.testclient import TestClient
    from test_reid import load_app

    main = load_app(monkeypatch, tmp_path, "test-tone-backfill")

    # After load_app: it reimports the app package, so patching before this would decorate a
    # class object the endpoint no longer uses.
    from app.db.session import SessionLocal
    from app.models.media import Image, PersonCrop
    from app.services.appearance_attributes import (
        AppearanceAttributeService as LoadedService,
    )

    # Both readings come off one pose run now, so the stub sits where that run happens.
    monkeypatch.setattr(
        LoadedService,
        "_read",
        lambda self, path: (ClothingTone("light", "dark", 20.0, 30.0), BodyShape(facing="front")),
    )

    described = {"object_type": "person", "appearance": {"gender": "male"}, "source": "vlm"}
    with TestClient(main.create_app()) as client:
        # The endpoint checks the file exists before reading it, so the rows need real files.
        from app.config.settings import get_settings

        crops_dir = get_settings().data_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for name in ("plain", "rich"):
            (crops_dir / f"{name}.png").write_bytes(b"stub")
        with SessionLocal() as db:
            image = Image(image_url="/data/frames/f.jpg", source_type="stream_frame")
            db.add(image)
            db.commit()
            db.refresh(image)
            ids = []
            for name, attributes in (("plain", None), ("rich", described)):
                crop = PersonCrop(
                    image_id=image.id,
                    crop_url=f"/data/crops/{name}.png",
                    bbox={"label": "person"},
                    attributes=attributes,
                )
                db.add(crop)
                db.commit()
                db.refresh(crop)
                ids.append(crop.id)

        result = client.post("/api/attributes/person-crops/tone-backfill?limit=10").json()

        with SessionLocal() as db:
            filled = db.get(PersonCrop, ids[0])
            untouched = db.get(PersonCrop, ids[1])

    assert result["updated"] == 1
    assert result["skipped_described"] == 1
    assert filled.attributes["clothing"] == {"upper_color": "light", "lower_color": "dark"}
    assert filled.attributes["facing"] == "front"
    assert untouched.attributes == described, "a VLM reading was overwritten by brightness"


# --- facing, from the shoulders ---------------------------------------------------------


def points_with(**named):
    """A 17-point skeleton with only the named joints placed; the rest read as missing."""

    names = {
        "nose": 0, "l_ear": 3, "r_ear": 4,
        "l_shoulder": 5, "r_shoulder": 6,
        "l_elbow": 7, "r_elbow": 8, "l_wrist": 9, "r_wrist": 10,
        "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14,
        "l_ankle": 15, "r_ankle": 16,
    }
    xy = np.zeros((17, 2), dtype="float64")
    conf = np.zeros(17, dtype="float64")
    for name, position in named.items():
        xy[names[name]] = position
        conf[names[name]] = 0.9
    return xy, conf


def seen_from(conf):
    return lambda index: conf[index] >= 0.5


def test_a_body_facing_the_camera_puts_its_left_shoulder_on_the_image_right(service):
    xy, conf = points_with(l_shoulder=(70, 40), r_shoulder=(30, 40))

    assert service._facing(xy, seen_from(conf)) == "front"


def test_the_shoulder_order_flips_when_the_body_turns_away(service):
    xy, conf = points_with(l_shoulder=(30, 40), r_shoulder=(70, 40))

    assert service._facing(xy, seen_from(conf)) == "back"


def test_stacked_shoulders_are_no_direction_at_all(service):
    """Edge-on, or shot from so far overhead that the projection carries nothing."""

    xy, conf = points_with(l_shoulder=(50, 30), r_shoulder=(52, 70))

    assert service._facing(xy, seen_from(conf)) is None


def test_a_missing_shoulder_is_not_a_direction(service):
    xy, conf = points_with(l_shoulder=(70, 40))

    assert service._facing(xy, seen_from(conf)) is None


# --- sleeve and trouser length ----------------------------------------------------------


def skin_crop(cv2, path, forearm=None, shin=None):
    """A grey body with a skin-toned face, and optionally bare forearm or shin patches."""

    # Grain, because a perfectly flat rectangle reads as motion blur to the sharpness gate and
    # every length would be withheld -- a property of the fixture, not of the extractor.
    image = np.full((200, 100, 3), 100, np.uint8)
    image = np.clip(
        image.astype("int16") + np.random.default_rng(7).integers(-25, 26, image.shape), 0, 255
    ).astype(np.uint8)
    skin = (110, 150, 190)  # BGR, a plausible face
    image[10:30, 40:60] = skin
    if forearm is not None:
        x, y = forearm
        image[y - 12 : y + 12, x - 12 : x + 12] = skin
    if shin is not None:
        x, y = shin
        image[y - 12 : y + 12, x - 12 : x + 12] = skin
    assert cv2.imwrite(str(path), image)
    return image


def test_a_bare_forearm_reads_as_a_short_sleeve(service, tmp_path):
    import cv2

    path = tmp_path / "arm.png"
    # Elbow to wrist runs down the outside of the body, clear of the torso box.
    image = skin_crop(cv2, path, forearm=(88, 82))
    xy, conf = points_with(
        nose=(50, 20), l_shoulder=(60, 50), r_shoulder=(40, 50),
        l_hip=(58, 110), r_hip=(42, 110), l_elbow=(88, 70), l_wrist=(88, 100),
    )

    assert service._shape(cv2, image, (xy, conf)).upper_length == "short"


def test_no_skin_on_the_arm_claims_nothing(service, tmp_path):
    """A hanging hand, a folded arm and a sleeve all look the same here, so none of them speak."""

    import cv2

    path = tmp_path / "covered.png"
    image = skin_crop(cv2, path)
    xy, conf = points_with(
        nose=(50, 20), l_shoulder=(60, 50), r_shoulder=(40, 50),
        l_hip=(58, 110), r_hip=(42, 110), l_elbow=(88, 70), l_wrist=(88, 100),
    )

    assert service._shape(cv2, image, (xy, conf)).upper_length is None


def test_an_arm_folded_across_the_chest_is_refused_rather_than_read(service, tmp_path):
    """The sample would land on the shirt, and reading it there is where "long" went wrong."""

    import cv2

    path = tmp_path / "folded.png"
    image = skin_crop(cv2, path, forearm=(50, 82))
    xy, conf = points_with(
        nose=(50, 20), l_shoulder=(60, 50), r_shoulder=(40, 50),
        l_hip=(58, 110), r_hip=(42, 110), l_elbow=(50, 70), l_wrist=(50, 100),
    )

    assert service._shape(cv2, image, (xy, conf)).upper_length is None


def test_a_covered_shin_reads_as_long_trousers(service, tmp_path):
    import cv2

    path = tmp_path / "legs.png"
    image = skin_crop(cv2, path)
    xy, conf = points_with(
        nose=(50, 20), l_shoulder=(60, 50), r_shoulder=(40, 50),
        l_knee=(55, 120), l_ankle=(55, 180),
    )

    assert service._shape(cv2, image, (xy, conf)).lower_length == "long"


def test_a_bare_shin_reads_as_shorts(service, tmp_path):
    import cv2

    path = tmp_path / "shorts.png"
    image = skin_crop(cv2, path, shin=(55, 144))
    xy, conf = points_with(
        nose=(50, 20), l_shoulder=(60, 50), r_shoulder=(40, 50),
        l_knee=(55, 120), l_ankle=(55, 180),
    )

    assert service._shape(cv2, image, (xy, conf)).lower_length == "short"


def test_length_is_withheld_on_a_blurred_crop_but_facing_is_not(service, tmp_path):
    """Blur erases the skin-to-cloth edge; it does not move the shoulders."""

    import cv2

    path = tmp_path / "blurred.png"
    image = skin_crop(cv2, path, shin=(55, 144))
    smeared = cv2.GaussianBlur(image, (31, 31), 0)
    xy, conf = points_with(
        nose=(50, 20), l_shoulder=(60, 50), r_shoulder=(40, 50),
        l_knee=(55, 120), l_ankle=(55, 180),
    )

    shape = service._shape(cv2, smeared, (xy, conf))

    assert shape.lower_length is None
    assert shape.facing == "front"


def test_a_face_the_model_cannot_see_leaves_length_unread(service, tmp_path):
    """Skin is judged against this person's own face, so without one there is no reference."""

    import cv2

    path = tmp_path / "headless.png"
    image = skin_crop(cv2, path, shin=(55, 144))
    xy, conf = points_with(
        l_shoulder=(60, 50), r_shoulder=(40, 50), l_knee=(55, 120), l_ankle=(55, 180),
    )

    assert service._shape(cv2, image, (xy, conf)).lower_length is None


def test_unread_attributes_are_absent_rather_than_null(service, tmp_path, monkeypatch):
    """A filter on sleeve length should skip a crop with no reading, not match a stated blank."""

    import cv2

    path = tmp_path / "body.png"
    assert cv2.imwrite(str(path), np.full((400, 100, 3), 128, np.uint8))
    monkeypatch.setattr(
        AppearanceAttributeService,
        "_keypoint_samples",
        lambda self, cv2_module, image, pose=None: (hsv(120, 200, 200), hsv(0, 5, 40)),
    )
    monkeypatch.setattr(AppearanceAttributeService, "_pose", lambda self, image: None)

    attributes = service.describe(path)

    assert "upper_length" not in attributes["clothing"]
    assert "facing" not in attributes


# --- the readings reaching search and the attribute panel -------------------------------


def test_only_the_attributes_that_were_read_reach_the_panel():
    """Every one of these abstains often, so a fixed row would mostly print 未知."""

    from app.services.observation_index import _chinese_attribute_labels

    labels = _chinese_attribute_labels(
        {
            "object_type": "person",
            "facing": "back",
            "clothing": {"upper_color": "dark", "lower_color": "light", "lower_length": "short"},
        }
    )

    assert labels["朝向"] == "背面"
    assert labels["裤长"] == "短裤/短裙"
    assert "袖长" not in labels
    assert labels["上衣颜色"] == "深色"


def test_short_and_long_mean_different_garments_on_top_and_bottom():
    from app.services.observation_index import _chinese_attribute_labels

    labels = _chinese_attribute_labels(
        {
            "object_type": "person",
            "clothing": {"upper_length": "short", "lower_length": "long"},
        }
    )

    assert labels["袖长"] == "短袖"
    assert labels["裤长"] == "长裤"


@pytest.mark.parametrize(
    ("query", "field", "value"),
    [
        ("穿短袖的人", "upper_length", "short"),
        ("短裤", "lower_length", "short"),
        ("长裤", "lower_length", "long"),
        ("背影", "facing", "back"),
        ("正面", "facing", "front"),
    ],
)
def test_a_length_or_direction_in_the_query_becomes_a_condition(query, field, value):
    from app.services.search import StructuredSearchService

    lengths = dict(StructuredSearchService._query_lengths(query))
    facing = StructuredSearchService._query_facing(query)
    found = {**lengths, **({"facing": facing} if facing else {})}

    assert found.get(field) == value


def test_a_length_condition_reads_the_nested_clothing_key():
    """The condition names a flat field; the extractor writes it under clothing."""

    from app.services.search import StructuredSearchService

    # Reading a path needs no collaborators, so the service is built without its dependencies.
    service = StructuredSearchService.__new__(StructuredSearchService)
    values = service._attribute_values({"clothing": {"upper_length": "short"}}, {}, "upper_length")

    assert values == ["short"]
