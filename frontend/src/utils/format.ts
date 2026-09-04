export const DISPLAY_TIME_ZONE = "Asia/Shanghai";

const dateTimeFormat = new Intl.DateTimeFormat("zh-CN", {
  timeZone: DISPLAY_TIME_ZONE,
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

const clockFormat = new Intl.DateTimeFormat("zh-CN", {
  timeZone: DISPLAY_TIME_ZONE,
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

export function fmtTime(value: string | Date | null | undefined): string {
  if (!value) return "无时间";
  return dateTimeFormat.format(new Date(value));
}

export function fmtClock(value: Date): string {
  return clockFormat.format(value);
}

export function shortId(value: string | null | undefined): string {
  return value ? value.slice(0, 8) : "-";
}

export function shortText(value: string | null | undefined, limit = 18): string {
  if (!value) return "";
  return value.length > limit ? `${value.slice(0, limit)}...` : value;
}

export function formatScore(value: unknown): string {
  const score = Number(value);
  return Number.isFinite(score) ? score.toFixed(2) : "-";
}

export function maskUrl(value: string): string {
  return value.replace(/:\/\/([^:/@]+):([^@]+)@/, "://$1:***@");
}
