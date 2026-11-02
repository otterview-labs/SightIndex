// Build-time presentation switch; the backend public-demo guard is the access boundary.
export const PUBLIC_DEMO = import.meta.env.VITE_PUBLIC_DEMO === "true";
