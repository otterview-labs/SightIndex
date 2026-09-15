import type { components } from "./schema";

type Schemas = components["schemas"];

export type VideoStream = Schemas["VideoStreamRead"];
export type VideoStreamCreate = Schemas["VideoStreamCreate"];
export type CountingLineConfig = Schemas["CountingLineConfig"];
export type CountSummary = Schemas["CountSummary"];
export type StreamActionResponse = Schemas["StreamActionResponse"];
export type VideoProcessResponse = Schemas["VideoProcessResponse"];

export type ImageRead = Schemas["ImageRead"];
export type PersonCropRead = Schemas["PersonCropRead"];

export type SearchFilters = Schemas["SearchFilters"];
export type SearchResultItem = Schemas["SearchResultItem"];
export type SearchResponse = Schemas["SearchResponse"];
export type VisualSearchRequest = Schemas["VisualSearchRequest"];
export type IndexRebuildResponse = Schemas["IndexRebuildResponse"];

export interface SemanticSearchRequest {
  query: string;
  top_k?: number;
  filters?: SearchFilters;
}

export type SemanticSearchItem = SearchResultItem & {
  match_type: "semantic_candidate";
  duplicate_crop_ids: string[];
};

export interface SemanticSearchResponse {
  mode: "semantic";
  items: SemanticSearchItem[];
  model: string;
  min_score: number;
  notice: string;
  scope_crops: number;
  indexed_scope_crops: number;
  candidates_examined: number;
  shortlist_limited: boolean;
}

export interface SemanticSearchStatus {
  enabled: boolean;
  configured: boolean;
  model: string;
  min_score: number;
  total_crops: number;
  indexed_crops: number;
  labeled_crops: number;
  attributes_enabled: boolean;
  auto_index_on_ingest: boolean;
}

export type ObservationIndexItem = Schemas["ObservationIndexItem"];
export type ObservationIndexResponse = Schemas["ObservationIndexResponse"];

export type Person = Schemas["PersonRead"];
export type PersonCreate = Schemas["PersonCreate"];
export type PersonUpdate = Schemas["PersonUpdate"];
export type PersonVisitStats = Schemas["PersonVisitStats"];
export type FaceEmbeddingRead = Schemas["FaceEmbeddingRead"];
export type FaceMatchItem = Schemas["FaceMatchItem"];
export type FaceRecognitionResponse = Schemas["FaceRecognitionResponse"];
export type FaceSearchResponse = Schemas["FaceSearchResponse"];
export type FaceDiagnosticItem = Schemas["FaceDiagnosticItem"];
export type FaceDiagnosticResponse = Schemas["FaceDiagnosticResponse"];
export type FaceRecognitionRebuildResponse = Schemas["FaceRecognitionRebuildResponse"];
export type PersonTrajectoryPoint = Schemas["PersonTrajectoryPoint"];
export type PersonTrajectoryResponse = Schemas["PersonTrajectoryResponse"];

export type ChatRequest = Schemas["ChatRequest"];
export type ChatResponse = Schemas["ChatResponse"];
export type ChatToolCall = Schemas["ChatToolCall"];

export type PersonCropAttributeResponse = Schemas["PersonCropAttributeResponse"];

export type ReidMatchItem = Schemas["ReidMatchItem"] & {
  attribute_evidence_weight?: number | null;
  attribute_conflict_weight?: number | null;
  face_reliability?: number | null;
  fusion_score?: number | null;
  evidence_level?: string | null;
  decision_reason?: string | null;
};
export type ReidSearchResponse = Schemas["ReidSearchResponse"];
export type ReidFaceCoverage = Schemas["ReidFaceCoverage"];
export type ReidCameraLink = Schemas["ReidCameraLink"] & {
  attribute_evidence_weight?: number | null;
  attribute_conflict_weight?: number | null;
  face_reliability?: number | null;
  fusion_score?: number | null;
  evidence_level?: string | null;
  decision_reason?: string | null;
};
export type ReidLinkResponse = Schemas["ReidLinkResponse"];
export type ReidStatusResponse = Schemas["ReidStatusResponse"];
export type ReidRebuildResponse = Schemas["ReidRebuildResponse"];
export type ReidFeedbackRead = Schemas["ReidFeedbackRead"];
export type ReidFeedbackUpsert = Schemas["ReidFeedbackUpsert"];

export type AttributeMap = Record<string, unknown>;

// openapi-typescript marks properties with a default as required; for request bodies they are not.
export type RequestBody<T, Required extends keyof T> = Pick<T, Required> & Partial<Omit<T, Required>>;
