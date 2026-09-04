import { api, jsonBody, queryString } from "./http";
import type {
  ChatRequest,
  ChatResponse,
  CountingLineConfig,
  FaceDiagnosticResponse,
  FaceEmbeddingRead,
  FaceRecognitionRebuildResponse,
  FaceRecognitionResponse,
  ImageRead,
  IndexRebuildResponse,
  ObservationIndexResponse,
  Person,
  PersonCreate,
  RequestBody,
  PersonCropRead,
  PersonTrajectoryResponse,
  ReidRebuildResponse,
  ReidLinkResponse,
  ReidSearchResponse,
  ReidStatusResponse,
  SearchResponse,
  StreamActionResponse,
  VideoProcessResponse,
  VideoStream,
  VideoStreamCreate,
  VisualSearchRequest,
} from "./types";

export interface StreamCounts {
  stream_id: string;
  stream_name: string;
  counting_event_count: number;
  total_counting_event_count: number;
  start_time: string | null;
  end_time: string | null;
}

export interface MediaCounts {
  image_with_crops_count: number;
  person_crop_count: number;
}

export const streams = {
  list: () => api<VideoStream[]>("/api/streams"),
  create: (payload: RequestBody<VideoStreamCreate, "name" | "stream_url">) =>
    api<VideoStream>("/api/streams", jsonBody(payload)),
  remove: (id: string) =>
    api<unknown>(`/api/streams/${id}`, { method: "DELETE" }),
  start: (id: string) =>
    api<StreamActionResponse>(`/api/streams/${id}/start`, { method: "POST" }),
  stop: (id: string) =>
    api<StreamActionResponse>(`/api/streams/${id}/stop`, { method: "POST" }),
  counts: (id: string) => api<StreamCounts>(`/api/streams/${id}/counts`),
  snapshot: (id: string) =>
    api<ImageRead>(`/api/streams/${id}/snapshot`, { method: "POST" }),
  saveCountingLine: (id: string, countingLine: CountingLineConfig | null) =>
    api<VideoStream>(`/api/streams/${id}/counting-line`, {
      method: "PATCH",
      body: JSON.stringify({ counting_line: countingLine }),
    }),
};

export const images = {
  listWithCrops: (limit = 60, offset = 0) =>
    api<ImageRead[]>(`/api/images${queryString({ limit, offset, has_crops: true })}`),
  get: (id: string) => api<ImageRead>(`/api/images/${id}`),
  process: (id: string) =>
    api<PersonCropRead[]>(`/api/images/${id}/process`, { method: "POST" }),
  upload: (body: FormData) =>
    api<ImageRead>("/api/images/upload", { method: "POST", body }),
};

export const crops = {
  list: (limit = 80, offset = 0) =>
    api<PersonCropRead[]>(`/api/person-crops${queryString({ limit, offset })}`),
  get: (cropId: string) => api<PersonCropRead>(`/api/person-crops/${cropId}`),
};

export const media = {
  counts: () => api<MediaCounts>("/api/media/counts"),
};

export const videos = {
  upload: (
    body: FormData,
    params: Record<string, string | number | boolean | undefined>,
  ) =>
    api<VideoProcessResponse>(`/api/videos/upload${queryString(params)}`, {
      method: "POST",
      body,
    }),
};

export const search = {
  personCrops: (payload: RequestBody<VisualSearchRequest, "query">) =>
    api<SearchResponse>("/api/search/person-crops", jsonBody(payload)),
  observations: (params: URLSearchParams) =>
    api<ObservationIndexResponse>(
      `/api/search/observations?${params.toString()}`,
    ),
  rebuildObservations: (limit = 500) =>
    api<IndexRebuildResponse>(
      `/api/search/observations/rebuild${queryString({ limit })}`,
      {
        method: "POST",
      },
    ),
};

export interface AttributeBackfillResult {
  requested: number;
  force: boolean;
  seen: number;
  updated: number;
  errors: string[];
}

export const attributes = {
  backfillCrops: (limit = 50, force = true) =>
    api<AttributeBackfillResult>(
      `/api/attributes/person-crops/backfill${queryString({ limit, force })}`,
      { method: "POST" },
    ),
};

export const persons = {
  list: (limit?: number) =>
    api<Person[]>(`/api/persons${queryString({ limit })}`),
  create: (payload: RequestBody<PersonCreate, "name">) =>
    api<Person>("/api/persons", jsonBody(payload)),
  faces: (personId: string) =>
    api<FaceEmbeddingRead[]>(`/api/persons/${personId}/faces`),
  addFace: (personId: string, body: FormData) =>
    api<FaceEmbeddingRead>(`/api/persons/${personId}/faces`, {
      method: "POST",
      body,
    }),
  enrollFromCrop: (personId: string, cropId: string) =>
    api<FaceEmbeddingRead>(
      `/api/persons/${personId}/faces/from-crop/${cropId}`,
      {
        method: "POST",
      },
    ),
  // Labelling a crop needs no face, so it works with face recognition switched off.
  labelledCrops: (personId: string) =>
    api<PersonCropRead[]>(`/api/persons/${personId}/crops`),
  labelCrop: (personId: string, cropId: string) =>
    api<PersonCropRead>(`/api/persons/${personId}/crops/${cropId}`, {
      method: "POST",
    }),
  unlabelCrop: (personId: string, cropId: string) =>
    api<PersonCropRead>(`/api/persons/${personId}/crops/${cropId}`, {
      method: "DELETE",
    }),
  trajectory: (personId: string, mode: string, signal?: AbortSignal) =>
    api<PersonTrajectoryResponse>(
      `/api/persons/${personId}/trajectory${queryString({
        limit: 100,
        mode,
        backfill_missing: false,
      })}`,
      { signal },
    ),
};

export const face = {
  recognize: (body: FormData, threshold: number) =>
    api<FaceRecognitionResponse>(
      `/api/face/recognize${queryString({ threshold })}`,
      {
        method: "POST",
        body,
      },
    ),
  diagnostics: (limit: number) =>
    api<FaceDiagnosticResponse>(
      `/api/face/diagnostics/recent${queryString({ limit })}`,
    ),
  rebuildRecognition: (limit = 1000) =>
    api<FaceRecognitionRebuildResponse>(
      `/api/face/index/rebuild${queryString({ limit })}`,
      {
        method: "POST",
      },
    ),
};

export const reid = {
  status: () => api<ReidStatusResponse>("/api/reid/status"),
  search: (body: FormData, topK?: number, signal?: AbortSignal) =>
    api<ReidSearchResponse>(`/api/reid/search${queryString({ top_k: topK })}`, {
      method: "POST",
      body,
      signal,
    }),
  // Best candidate at each other camera, whatever its score -- ranking, not a verdict.
  links: (cropId: string) =>
    api<ReidLinkResponse>(`/api/reid/crops/${cropId}/links`, { method: "POST" }),
  similarToCrop: (cropId: string, topK?: number) =>
    api<ReidSearchResponse>(
      `/api/reid/crops/${cropId}/similar${queryString({ top_k: topK })}`,
      { method: "POST" },
    ),
  rebuild: (limit = 200) =>
    api<ReidRebuildResponse>(`/api/reid/index/rebuild${queryString({ limit })}`, {
      method: "POST",
    }),
};

export const chat = {
  ask: (payload: ChatRequest) =>
    api<ChatResponse>("/api/chat", jsonBody(payload)),
};
