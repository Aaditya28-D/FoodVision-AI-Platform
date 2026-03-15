export function formatMs(value) {
  if (value == null) return "-";
  return `${Number(value).toFixed(1)} ms`;
}

export function formatConfidence(value) {
  if (value == null) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

export function formatSimilarity(value) {
  if (value == null) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export function prettifyLabel(value) {
  if (!value) return "-";
  return value.replaceAll("_", " ");
}

export function prettifyModelName(value) {
  if (!value) return "-";
  return value.replaceAll("_", " ");
}

export function getPredictionSourceBadge(modelName) {
  if (!modelName) {
    return {
      text: "Prediction source unavailable",
      className: "border-slate-200 bg-slate-50 text-slate-700",
    };
  }

  if (modelName.startsWith("smart_router->")) {
    const routedModel = modelName.split("->")[1] || modelName;
    return {
      text: `Powered by smart routing (${prettifyModelName(routedModel)})`,
      className: "border-violet-200 bg-violet-50 text-violet-700",
    };
  }

  if (modelName.startsWith("ensemble")) {
    return {
      text: "Powered by ensemble",
      className: "border-indigo-200 bg-indigo-50 text-indigo-700",
    };
  }

  return {
    text: `Powered by ${prettifyModelName(modelName)}`,
    className: "border-slate-200 bg-slate-50 text-slate-700",
  };
}
