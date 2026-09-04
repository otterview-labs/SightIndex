import type { AttributeMap } from "@/api/types";

const COLOR_LABELS: Record<string, string> = {
  white: "白色",
  black: "黑色",
  red: "红色",
  yellow: "黄色",
  blue: "蓝色",
  green: "绿色",
  gray: "灰色",
  grey: "灰色",
  brown: "棕色",
  orange: "橙色",
  purple: "紫色",
  pink: "粉色",
  // What the pixel reader falls back to when saturation is too low to name a hue, which is
  // most of this footage. Without these it rendered the raw English into the chip.
  light: "浅色",
  dark: "深色",
};

const LENGTH_LABELS: Record<string, Record<string, string>> = {
  upper: { short: "短袖", long: "长袖" },
  lower: { short: "短裤/短裙", long: "长裤" },
};

const FACING_LABELS: Record<string, string> = { front: "正面", back: "背面" };

const STATURE_LABELS: Record<string, string> = { tall: "偏高", average: "中等", short: "偏矮" };

const HAIR_LABELS: Record<string, string> = {
  bald: "光头",
  shaved: "寸头",
  short_hair: "短发",
  long_hair: "长发",
};

export function colorLabel(value: string): string {
  return COLOR_LABELS[value] ?? value;
}

export function hairLabel(value: string): string {
  return HAIR_LABELS[value] ?? value;
}

export function normalizedAttribute(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value).trim().toLowerCase();
  return text && text !== "unknown" && text !== "null" ? text : "";
}

export function truthyAttribute(value: unknown): boolean {
  return value === true || String(value).trim().toLowerCase() === "true";
}

export interface AttributeChip {
  label: string;
  value: string;
}

const MAX_CHIPS = 5;

function section(attributes: AttributeMap, key: string): AttributeMap {
  const value = attributes[key];
  return value && typeof value === "object" ? (value as AttributeMap) : {};
}

export function structuredAttributeChips(attributes: unknown): AttributeChip[] {
  if (!attributes || typeof attributes !== "object") return [];
  const map = attributes as AttributeMap;
  const appearance = section(map, "appearance");
  const clothing = section(map, "clothing");
  const objects = section(map, "objects");
  const behavior = section(map, "behavior");

  const chips: AttributeChip[] = [];
  const push = (label: string, value: string) => chips.push({ label, value });

  const upperColor = normalizedAttribute(map.top_color ?? clothing.upper_color);
  const lowerColor = normalizedAttribute(map.bottom_color ?? clothing.lower_color);
  if (upperColor) push("上衣", colorLabel(upperColor));
  if (lowerColor) push("下装", colorLabel(lowerColor));
  if (truthyAttribute(map.has_backpack ?? objects.backpack)) push("物品", "背包");
  if (truthyAttribute(map.has_glasses ?? appearance.glasses)) push("外观", "眼镜");
  if (truthyAttribute(map.has_hat ?? appearance.hat)) push("外观", "帽子");
  if (truthyAttribute(objects.holding_phone ?? behavior.looking_at_phone)) push("行为", "玩手机");
  if (truthyAttribute(behavior.smoking ?? objects.cigarette)) push("行为", "抽烟");
  if (truthyAttribute(behavior.falling ?? behavior.lying_on_ground)) push("行为", "跌倒");
  if (truthyAttribute(behavior.fighting ?? behavior.physical_conflict)) push("行为", "打架");
  const hair = normalizedAttribute(appearance.hair ?? map.hair);
  if (hair && hair !== "unknown") push("发型", hairLabel(hair));
  const upperLength = normalizedAttribute(clothing.upper_length);
  const lowerLength = normalizedAttribute(clothing.lower_length);
  if (upperLength) push("袖长", LENGTH_LABELS.upper[upperLength] ?? upperLength);
  if (lowerLength) push("裤长", LENGTH_LABELS.lower[lowerLength] ?? lowerLength);
  const facing = normalizedAttribute(map.facing);
  if (facing) push("朝向", FACING_LABELS[facing] ?? facing);
  const stature = normalizedAttribute(section(map, "stature").band);
  if (stature) push("身高", STATURE_LABELS[stature] ?? stature);

  return chips.slice(0, MAX_CHIPS);
}
