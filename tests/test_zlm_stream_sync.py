from scripts.sync_zlm_streams import candidate_from_media_item


def test_candidate_from_media_item_rewrites_tunneled_h264_url() -> None:
    item = {
        "originUrl": (
            "rtsp://127.0.0.1:7003/device/"
            "00000000000000000001/00000000000000000002"
            "?_type=device&token=abc&app=ai&streamId=a/b/c"
        ),
        "tracks": [
            {"codec_type": 1, "codec_id_name": "PCMA"},
            {
                "codec_type": 0,
                "codec_id_name": "H264",
                "width": 640,
                "height": 480,
                "fps": 10.0,
            },
        ],
    }

    candidate = candidate_from_media_item(
        item,
        source_host="127.0.0.1:7003",
        target_host="127.0.0.1:17003",
        prefix="ZLM-H264-",
    )

    assert candidate is not None
    assert candidate.name == "ZLM-H264-00000000000000000002"
    assert candidate.channel_id == "00000000000000000002"
    assert "127.0.0.1:17003" in candidate.stream_url
    assert "token=abc" in candidate.stream_url
    assert candidate.width == 640
    assert candidate.height == 480


def test_candidate_from_media_item_ignores_non_h264_or_missing_token() -> None:
    base_item = {
        "originUrl": (
            "rtsp://127.0.0.1:7003/device/"
            "00000000000000000003/00000000000000000004"
            "?_type=device&token=abc"
        ),
        "tracks": [{"codec_type": 0, "codec_id_name": "H265"}],
    }

    assert (
        candidate_from_media_item(
            base_item,
            source_host="127.0.0.1:7003",
            target_host="127.0.0.1:17003",
            prefix="ZLM-H264-",
        )
        is None
    )

    missing_token = {
        **base_item,
        "originUrl": base_item["originUrl"].replace("token=abc", "app=ai"),
        "tracks": [{"codec_type": 0, "codec_id_name": "H264"}],
    }
    assert (
        candidate_from_media_item(
            missing_token,
            source_host="127.0.0.1:7003",
            target_host="127.0.0.1:17003",
            prefix="ZLM-H264-",
        )
        is None
    )
