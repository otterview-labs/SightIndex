from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    app_name: str = "SightIndex"
    environment: str = "dev"
    local_timezone: str = "Asia/Shanghai"
    database_url: str = "sqlite:///./sightindex.db"
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=200)
    database_pool_timeout_seconds: int = Field(default=10, ge=1, le=120)
    database_pool_recycle_seconds: int = Field(default=1800, ge=30, le=86400)
    data_dir: Path = Path("data")
    media_retention_days: int = Field(default=30, ge=1, le=3660)
    public_base_url: str = "http://localhost:8000"
    auto_create_tables: bool = True
    app_basic_auth_username: str | None = None
    app_basic_auth_password: str | None = None
    face_match_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    face_embedding_provider: str = "insightface"
    face_embedding_dim: int = Field(default=512, ge=1)
    face_embedding_device: str | None = None
    face_insightface_model: str = "buffalo_l"
    face_insightface_root: Path | None = None
    # 1280 makes the SCRFD detector return nothing at every frame size we tested; 640 is
    # InsightFace's own default and detects reliably from 180x220 crops up to 1080p frames.
    face_insightface_det_size: int = Field(default=640, ge=128, le=2048)
    # InsightFace fetches a 281MB bundle on first use, inline in whichever request touched it.
    # On a slow link that pins a worker thread and its database session for hours -- observed at
    # 10KB/s, which is an eight-hour request, and under SQLite a long-held read blocks ingest
    # writes. Seed the model deliberately, the same way the YOLO pose weights are.
    face_insightface_allow_download: bool = False
    face_candidate_upscale_min_width: int = Field(default=480, ge=0, le=4096)
    face_candidate_upscale_min_height: int = Field(default=720, ge=0, le=4096)
    face_candidate_upscale_max_factor: float = Field(default=3.0, ge=1.0, le=6.0)
    face_recognition_on_ingest: bool = True
    face_fallback_to_full_image: bool = True
    face_max_library_scan: int = Field(default=5000, ge=1, le=100000)
    face_library_cache_enabled: bool = True
    face_library_cache_ttl_seconds: int = Field(default=60, ge=1, le=3600)
    count_dedup_seconds: int = Field(default=60, ge=1)
    line_crossing_point: str = "bottom_center"
    line_crossing_match_distance: float = Field(default=0.32, ge=0.01, le=1.0)
    # One crop per visit rather than one per capture interval. Without this a person lingering
    # at a door is stored every frame -- 127 times over five minutes on this deployment -- and
    # every copy is embedded, indexed and returned, crowding everyone else out of the results.
    # Clothing tone straight off the pixels, for deployments with no VLM. It names a hue only
    # when saturation justifies one; below the floor it reports brightness rather than inventing
    # a colour, because on this footage the median saturation is 29 and a hue there is noise.
    appearance_tone_on_ingest: bool = True
    # A hue is claimed only above this saturation and brightness. The floor sits at the 90th
    # percentile of measured garment patches: below it, every strong colour in this footage came
    # from shadow, where hue is noise -- that is what turned a black shirt blue.
    appearance_tone_saturation_floor: float = Field(default=110.0, ge=0.0, le=255.0)
    appearance_tone_hue_value_floor: float = Field(default=90.0, ge=0.0, le=255.0)
    # Light and dark are judged against the crop's own exposure, which ranges 58..184 across
    # these cameras. An absolute cut called white shirts in a shaded doorway dark.
    appearance_tone_dark_ratio: float = Field(default=0.90, ge=0.1, le=2.0)
    person_crop_dedupe_enabled: bool = True
    # Normalised to the frame, so the same value holds at any resolution.
    person_crop_visit_match_distance: float = Field(default=0.32, ge=0.01, le=1.0)
    # No sighting for this long ends the visit; the next one is stored again.
    person_crop_visit_idle_seconds: float = Field(default=6.0, ge=0.5, le=600.0)
    # A visit that never ends is stored again anyway, so someone standing at the door all
    # afternoon does not appear exactly once at the moment they arrived.
    person_crop_visit_max_seconds: float = Field(default=300.0, ge=5.0, le=86400.0)
    person_detector: str = "yolo"
    hog_hit_threshold: float = Field(default=0.5, ge=-2.0, le=5.0)
    yolo_model: str = "yolo11n.pt"
    yolo_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    yolo_image_size: int = Field(default=640, ge=320, le=1920)
    yolo_device: str | None = None
    yolo_service_url: str = "http://127.0.0.1:19121"
    yolo_service_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_service_iou: float = Field(default=0.5, ge=0.0, le=1.0)
    yolo_service_image_size: int = Field(default=1280, ge=64, le=2560)
    yolo_service_max_det: int = Field(default=100, ge=1, le=1000)
    yolo_service_timeout_seconds: int = Field(default=120, ge=1)
    frame_jpeg_quality: int = Field(default=95, ge=30, le=100)
    thumbnail_jpeg_quality: int = Field(default=92, ge=30, le=100)
    person_crop_jpeg_quality: int = Field(default=96, ge=30, le=100)
    person_crop_min_bbox_width: int = Field(default=0, ge=0)
    person_crop_min_bbox_height: int = Field(default=0, ge=0)
    # Below this the detector is mostly finding floor reflections and white objects, and those
    # crops do real harm rather than merely wasting space: blank patches resemble each other, so
    # two of them matched at 0.710 across cameras -- higher than any pair of actual people, and
    # they surfaced as the top result. 37% of a day's crops sat under 0.70 here.
    person_crop_min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    # Doorway cameras often devote most pixels to the room rather than the person. Store a
    # display-friendly crop while keeping the scale bounded: this cannot invent detail, but
    # Lanczos resampling plus edge-aware sharpening makes the detail the camera did capture much
    # easier to judge in the operator UI and more reliable for small-face detection.
    person_crop_upscale_min_width: int = Field(default=320, ge=0)
    person_crop_upscale_min_height: int = Field(default=720, ge=0)
    person_crop_upscale_max_factor: float = Field(default=2.0, ge=1.0, le=4.0)
    person_crop_sharpen_amount: float = Field(default=0.32, ge=0.0, le=1.0)
    # Ignore tiny high-frequency differences so sharpening does not turn sensor/JPEG noise into
    # false clothing texture. Zero preserves the legacy full-frame unsharp mask.
    person_crop_sharpen_threshold: int = Field(default=4, ge=0, le=64)
    person_crop_padding_x_ratio: float = Field(default=0.24, ge=0.0, le=1.0)
    person_crop_padding_top_ratio: float = Field(default=0.55, ge=0.0, le=1.0)
    person_crop_padding_bottom_ratio: float = Field(default=0.16, ge=0.0, le=1.0)
    stream_store_empty_frames: bool = False
    stream_autostart_running: bool = True
    rtsp_transport: str = "tcp"
    rtsp_open_timeout_ms: int = Field(default=8000, ge=100, le=60000)
    rtsp_read_timeout_ms: int = Field(default=8000, ge=100, le=60000)
    stream_warmup_frames: int = Field(default=3, ge=0, le=60)
    stream_corrupt_frame_mean_diff_threshold: float = Field(default=45.0, ge=0.0, le=255.0)
    stream_diagnostics_enabled: bool = False
    stream_diagnostics_interval_seconds: float = Field(default=10.0, ge=0.5, le=3600.0)
    stream_diagnostics_keep_latest_frame: bool = False
    milvus_enabled: bool = False
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str | None = None
    milvus_password: str | None = None
    milvus_db: str = "default"
    # Stable logical identity of the Milvus database that owns the vectors. When omitted,
    # the client derives one from host/port/db. Set this explicitly when the same database
    # moves endpoints so SQL markers keep referring to the same vector namespace.
    milvus_namespace_id: str | None = None
    milvus_collection_prefix: str = "sightindex"
    # Optional separate namespace for CLIP/Qwen image and person-crop vectors.
    # ReID and face collections keep the stable global prefix so changing a
    # visual model does not silently detach an existing identity index.
    milvus_visual_collection_prefix: str | None = None
    milvus_metric_type: str = "COSINE"
    milvus_timeout_seconds: float = Field(default=3.0, ge=0.1, le=60.0)
    # Flush allocates and seals segments, which on a freshly created collection takes far
    # longer than a search. Sharing the 3s search timeout makes the very first index of a new
    # deployment fail, every time.
    milvus_flush_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    milvus_failure_cooldown_seconds: int = Field(default=60, ge=0, le=3600)
    embedding_provider: str = "none"
    embedding_dim: int = Field(default=2560, ge=1)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    visual_embedding_provider: str = "none"
    visual_embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    visual_embedding_dim: int = Field(default=512, ge=1)
    visual_embedding_device: str | None = None
    visual_embedding_instruction: str = "Retrieve images that match the user query."
    visual_embedding_service_url: str | None = None
    visual_embedding_service_api_key: str | None = None
    visual_embedding_service_timeout_seconds: int = Field(default=15, ge=1)
    visual_embedding_service_failure_cooldown_seconds: int = Field(
        default=60,
        ge=0,
        le=3600,
    )
    visual_embedding_max_concurrency: int = Field(default=1, ge=1, le=16)
    visual_embedding_queue_timeout_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    visual_search_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    qwen3_vl_embedding_repo_dir: Path | None = None
    qwen3_vl_embedding_pythonpath: str | None = None
    qwen3_vl_embedding_torch_dtype: str | None = "bfloat16"
    qwen3_vl_embedding_attn_implementation: str | None = None
    vlm_provider: str = "none"
    vlm_base_url: str = "http://127.0.0.1:8001/v1"
    vlm_api_key: str | None = None
    vlm_service_api_key: str | None = None
    vlm_model: str = "Qwen3.6-27B"
    vlm_timeout_seconds: int = Field(default=120, ge=1)
    vlm_max_tokens: int = Field(default=256, ge=1, le=4096)
    vlm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    vlm_caption_on_index: bool = False
    vlm_structured_on_ingest: bool = False
    vlm_structured_max_tokens: int = Field(default=1200, ge=1, le=4096)
    # Low-confidence VLM attributes are more harmful to search than missing values. Keep the
    # threshold configurable so each camera/model can trade recall for precision explicitly.
    vlm_structured_min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    vlm_rerank_enabled: bool = False
    vlm_rerank_service_url: str | None = None
    vlm_rerank_service_api_key: str | None = None
    vlm_rerank_provider: str = "none"
    vlm_rerank_model: str = "Qwen/Qwen3-VL-Reranker-2B"
    vlm_rerank_device: str | None = None
    vlm_rerank_candidate_limit: int = Field(default=30, ge=1, le=100)
    vlm_rerank_max_workers: int = Field(default=8, ge=1, le=64)
    vlm_rerank_min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vlm_rerank_timeout_seconds: int = Field(default=180, ge=1)
    embedding_rerank_enabled: bool = False
    embedding_rerank_candidate_limit: int = Field(default=50, ge=1, le=100)
    embedding_rerank_max_workers: int = Field(default=8, ge=1, le=64)

    # Person re-identification (SapiensID), served by deploy/agx/reid_service.
    reid_enabled: bool = False
    reid_service_url: str | None = None
    reid_service_api_key: str | None = None
    reid_model: str = "sapiensid_wb12m"
    # Composite SHA-256 of the backbone/config plus pose and DFA aligner assets for the official
    # wb12m pipeline currently deployed with SightIndex. The service derives this from the bytes
    # it actually loaded, so an asset replacement cannot silently reuse an incompatible index.
    reid_checkpoint_revision: str = (
        "sha256:6302aa80abdd5a3fba51b4d892dfdae515cfa0ca5f3762031890c63e6a688971"
    )
    reid_embedding_dim: int = Field(default=4096, ge=1)
    # Must match the service's PREPROCESS_VERSION; vectors across preprocessing versions are
    # not comparable, so this participates in the index fingerprint and response validation.
    reid_preprocess_version: str = "squarepad-v1"
    reid_timeout_seconds: int = Field(default=60, ge=1)
    reid_failure_cooldown_seconds: int = Field(default=30, ge=0)
    reid_max_concurrency: int = Field(default=2, ge=1, le=32)
    reid_queue_timeout_seconds: float = Field(default=20.0, ge=0.1)
    reid_index_on_ingest: bool = True
    reid_search_top_k: int = Field(default=20, ge=1, le=200)
    # Fetch deeply enough that a camera-domain shift cannot let one doorway occupy the entire
    # raw candidate pool before results are grouped. The live index count remains the upper bound.
    reid_candidate_pool_max: int = Field(default=5000, ge=200, le=16384)
    reid_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # The same person seen at another camera scores far lower: measured here, same-camera best
    # matches sit around 0.89 while the best cross-camera pair in 24378 temporally plausible
    # combinations reached 0.481. Holding both to one bar means a real crossing is never
    # returned, which is the one question the feature exists to answer.
    #
    # Calibrated against pairs that cannot be the same person -- captured at the two doors within
    # eight seconds, which nobody can walk. Over 1330 of those the score reaches 0.440 by chance
    # alone, and a single search compares the query against hundreds of cross-camera crops, so a
    # bar below that returns coincidences. 0.45 clears the whole null distribution: zero of the
    # 1330 known-different pairs pass it.
    #
    # The cost is recall, and it is real: only the strongest crossings clear it. That is the
    # honest trade on this footage, where a genuine crossing scores 0.43 to 0.48 and chance
    # reaches 0.44. Better crops would separate the two; a lower value here only blurs them.
    reid_min_score_cross_camera: float = Field(default=0.45, ge=0.0, le=1.0)
    # A camera whose best match ranks below twenty same-camera visits never appears at all.
    # These slots are taken from the most over-represented camera, so the ranking survives.
    # How far height may move a cross-camera match in the ranking. Cross-camera scores that mean
    # anything sit between 0.43 and 0.58, so a swing of 0.03 reorders candidates that ReID cannot
    # separate while leaving it unable to lift a 0.30 past a 0.48. Height is not strong enough
    # here to overrule appearance: measured on 15 cross-camera candidate pairs it agrees to 23
    # percentile points where random pairs differ by 34, real (bootstrap p=0.018) but modest, and
    # those pairs are not confirmed identities. Raise it once a walk-through provides ground truth.
    reid_stature_weight: float = Field(default=0.03, ge=0.0, le=0.5)
    # Stable structured appearance labels refine ReID candidates without changing the calibrated
    # embedding admission thresholds. A single conflict is tolerated and down-ranked because one
    # camera can misread colour/length under glare or occlusion; two independent high-confidence
    # conflicts reject a candidate. Missing or uncertain labels always degrade to vector-only.
    reid_attribute_filter_enabled: bool = True
    reid_attribute_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    reid_attribute_hard_conflicts: int = Field(default=2, ge=1, le=7)
    reid_attribute_hard_confidence: float = Field(default=1.6, ge=0.0, le=7.0)
    reid_attribute_full_weight: float = Field(default=3.0, ge=0.1, le=7.0)
    # Clothes can reject an otherwise plausible identity only during a nearby visit. Across days
    # they are expected to change, so their evidence remains a continuous ranking nudge.
    reid_attribute_hard_filter_window_hours: float = Field(default=4.0, ge=0.0, le=168.0)
    reid_attribute_weight: float = Field(default=0.04, ge=0.0, le=0.5)
    reid_attribute_analyze_upload: bool = True
    reid_attribute_enrich_query_crop: bool = True
    # Face evidence is evaluated only for a small, camera-balanced ReID shortlist. A verified
    # face match outranks body appearance, while no-face and low-quality cases remain body-only.
    reid_face_priority_enabled: bool = True
    reid_face_candidate_limit: int = Field(default=12, ge=1, le=100)
    reid_face_min_quality: float = Field(default=0.55, ge=0.0, le=1.0)
    reid_face_strong_reliability: float = Field(default=0.70, ge=0.0, le=1.0)
    # A score below this is not merely "unconfirmed"; both high-quality faces were measurable
    # and they are clearly different. Reject it from the match list instead of letting similar
    # clothing keep an acknowledged face mismatch alive. Scores between this and the ordinary
    # face-match threshold remain uncertain and are only demoted.
    reid_face_hard_reject_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    reid_camera_quota: int = Field(default=3, ge=0, le=50)
    # The highest cross-camera score reached by pairs that cannot be the same person -- captured
    # at two doors within eight seconds. Links below it are shown but marked as within reach of
    # coincidence, so a weak best-candidate is never mistaken for a finding.
    reid_chance_ceiling: float = Field(default=0.44, ge=0.0, le=1.0)
    # One person lingering at one door produces a frame every reid interval, and twenty of those
    # fill a top-20 completely -- burying the other camera, which is the only reason to run the
    # search. Consecutive hits from the same camera within this many seconds count as one visit.
    # 0 returns every frame.
    reid_collapse_window_seconds: float = Field(default=60.0, ge=0.0, le=3600.0)
    # Camera and time alone are not enough to call two frames one visit: at a busy door every
    # gap is under the window and everything chains together, strangers included. Frames must
    # also look like each other. Measured same-identity similarity runs 0.79-0.98 and
    # cross-identity tops out near 0.30, so the gap between them is wide. 0 skips the test.
    reid_collapse_identity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    # A person's gallery is many crops; score a candidate by the mean of its best few matches
    # rather than a single nearest neighbour, which is easily won by one lucky frame.
    reid_gallery_size: int = Field(default=8, ge=1, le=64)
    reid_gallery_top_k: int = Field(default=3, ge=1, le=32)
    # A crop selected from the observation table is only one instant of a visit. Build a small
    # query gallery from nearby, identity-consistent frames so blur, pose and a turned-away face
    # in that one instant do not decide the whole search. The source crop always remains a member;
    # failure to read vectors or find neighbours therefore degrades exactly to the old behaviour.
    reid_query_tracklet_frames: int = Field(default=6, ge=1, le=16)
    reid_query_tracklet_window_seconds: float = Field(default=30.0, ge=0.0, le=300.0)
    reid_query_tracklet_identity_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    reid_query_tracklet_candidate_limit: int = Field(default=80, ge=1, le=500)
    embedding_rerank_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    vlm_rerank_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    vlm_structured_prompt: str = (
        "你是监控图片结构化解析器。请只输出合法 JSON，不要输出 Markdown。"
        "所有枚举值使用英文小写。未知或看不清时填 unknown 或 null。"
        "置信度字段取 0 到 1。"
    )
    vlm_caption_prompt: str = (
        "你是监控图像检索标注器。请用中文生成一段适合向量检索的简洁描述，"
        "重点包含人物外观、衣服颜色、发型、眼镜、帽子、背包、姿态、场景。"
        "如果图中没有人，也要描述可见场景。不要编造看不见的信息。"
    )
    vector_index_on_ingest: bool = False
    vector_index_on_ingest_background: bool = True
    vector_index_background_max_queue: int = Field(default=5000, ge=1, le=100000)
    vector_index_background_batch_size: int = Field(default=8, ge=1, le=200)
    vector_index_background_idle_seconds: float = Field(default=0.5, ge=0.05, le=60.0)
    vector_index_background_max_retries: int = Field(default=3, ge=0, le=20)
    vector_index_background_retry_delay_seconds: float = Field(default=10.0, ge=0.0, le=3600.0)
    vector_index_lease_seconds: float = Field(default=300.0, ge=5.0, le=86400.0)
    person_trajectory_vector_enabled: bool = True
    person_trajectory_vector_min_score: float = Field(default=0.72, ge=0.0, le=1.0)
    person_trajectory_face_seed_vector_min_score: float = Field(default=0.40, ge=0.0, le=1.0)
    person_trajectory_vector_seed_limit: int = Field(default=3, ge=1, le=50)
    person_trajectory_vector_top_k: int = Field(default=40, ge=1, le=500)
    person_trajectory_vector_max_seconds: float = Field(default=8.0, ge=0.1, le=300.0)
    person_trajectory_vector_embedding_timeout_seconds: int = Field(default=5, ge=1, le=300)
    person_trajectory_reid_max_seconds: float = Field(default=8.0, ge=0.1, le=300.0)

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def videos_dir(self) -> Path:
        return self.data_dir / "videos"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def diagnostics_dir(self) -> Path:
        return self.data_dir / "diagnostics"


@lru_cache
def get_settings() -> Settings:
    return Settings()
