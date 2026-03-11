import { useMemo, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";

function formatMs(value) {
  if (value == null) return "-";
  return `${Number(value).toFixed(1)} ms`;
}

function formatConfidence(value) {
  if (value == null) return "-";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function SummaryCard({ summary }) {
  if (!summary) return null;

  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-xl backdrop-blur">
      <h2 className="mb-4 text-xl font-semibold text-white">Battle Summary</h2>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <div className="rounded-2xl bg-slate-900/70 p-4">
          <p className="text-sm text-slate-400">Fastest model</p>
          <p className="mt-1 text-lg font-semibold text-cyan-300">{summary.fastest_model}</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-4">
          <p className="text-sm text-slate-400">Highest confidence</p>
          <p className="mt-1 text-lg font-semibold text-emerald-300">{summary.highest_confidence_model}</p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-4">
          <p className="text-sm text-slate-400">Agreement</p>
          <p className="mt-1 text-lg font-semibold text-amber-300">
            {summary.all_models_agree ? "All agree" : "Disagreement"}
          </p>
        </div>
        <div className="rounded-2xl bg-slate-900/70 p-4">
          <p className="text-sm text-slate-400">Majority label</p>
          <p className="mt-1 text-lg font-semibold text-fuchsia-300">{summary.majority_label}</p>
        </div>
      </div>
    </div>
  );
}

function PredictionList({ predictions }) {
  return (
    <div className="space-y-2">
      {predictions.map((item, index) => (
        <div
          key={`${item.class_name}-${index}`}
          className="flex items-center justify-between rounded-xl bg-slate-900/60 px-3 py-2"
        >
          <div className="flex items-center gap-3">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-white/10 text-xs text-slate-300">
              {index + 1}
            </span>
            <span className="text-sm font-medium text-slate-100">{item.class_name}</span>
          </div>
          <span className="text-sm text-cyan-300">{formatConfidence(item.confidence)}</span>
        </div>
      ))}
    </div>
  );
}

function ModelBattleCard({ item }) {
  const topPrediction = item.comparison.top_prediction;
  const heatmapUrl = `${API_BASE_URL}${item.explanation.heatmap_url}`;

  return (
    <div className="overflow-hidden rounded-3xl border border-white/10 bg-white/5 shadow-xl backdrop-blur">
      <div className="border-b border-white/10 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-xl font-semibold text-white">{item.comparison.model_name}</h3>
            <p className="mt-1 text-sm text-slate-400">
              Device: <span className="text-slate-200">{item.comparison.device}</span>
            </p>
          </div>
          <div className="grid gap-2 text-right">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500">Top prediction</p>
              <p className="text-lg font-semibold text-emerald-300">{topPrediction.class_name}</p>
            </div>
            <div className="text-sm text-slate-300">
              {formatConfidence(topPrediction.confidence)} · {formatMs(item.comparison.inference_time_ms)}
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 p-5 xl:grid-cols-[1.05fr_0.95fr]">
        <div>
          <h4 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Top-{item.comparison.predictions.length} predictions
          </h4>
          <PredictionList predictions={item.comparison.predictions} />
        </div>

        <div>
          <h4 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Grad-CAM Explanation
          </h4>
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/70">
            <img
              src={heatmapUrl}
              alt={`${item.comparison.model_name} heatmap`}
              className="h-auto w-full object-contain"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [battleData, setBattleData] = useState(null);
  const [error, setError] = useState("");

  const preview = useMemo(() => previewUrl, [previewUrl]);

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setBattleData(null);
    setError("");

    if (file) {
      const localUrl = URL.createObjectURL(file);
      setPreviewUrl(localUrl);
    } else {
      setPreviewUrl("");
    }
  }

  async function handleAnalyze(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please choose an image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setBattleData(null);

      const formData = new FormData();
      formData.append("image", selectedFile);

      const response = await fetch(`${API_BASE_URL}/battle?top_k=${topK}`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Battle request failed.");
      }

      setBattleData(data);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen px-4 py-8 text-slate-100 md:px-8 xl:px-10">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8">
          <div className="inline-flex items-center rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium tracking-wide text-cyan-300">
            FoodVision AI Platform · Battle Mode
          </div>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-white md:text-5xl">
            Multi-Model Food Battle
          </h1>
          <p className="mt-3 max-w-3xl text-base leading-7 text-slate-300 md:text-lg">
            Upload one food image and compare EfficientNet-B0, ResNet-50, and MobileNetV3-Large
            across prediction confidence, speed, and Grad-CAM explanations.
          </p>
        </header>

        <div className="grid gap-8 xl:grid-cols-[380px_1fr]">
          <aside className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-xl backdrop-blur">
            <h2 className="text-xl font-semibold text-white">Upload Image</h2>

            <form onSubmit={handleAnalyze} className="mt-5 space-y-5">
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300">Food image</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="block w-full rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm text-slate-200 file:mr-4 file:rounded-xl file:border-0 file:bg-cyan-500 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-cyan-400"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium text-slate-300">Top-K predictions</label>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3 text-sm text-slate-200 outline-none focus:border-cyan-400"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-2xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Analyzing..." : "Run Battle Mode"}
              </button>
            </form>

            {error ? (
              <div className="mt-5 rounded-2xl border border-red-400/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </div>
            ) : null}

            <div className="mt-6">
              <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Preview
              </h3>
              <div className="overflow-hidden rounded-3xl border border-white/10 bg-slate-950/70">
                {preview ? (
                  <img src={preview} alt="Selected upload preview" className="h-auto w-full object-cover" />
                ) : (
                  <div className="flex h-72 items-center justify-center px-6 text-center text-sm text-slate-500">
                    Choose an image to preview it here.
                  </div>
                )}
              </div>
            </div>
          </aside>

          <main className="space-y-6">
            {battleData ? (
              <>
                <SummaryCard summary={battleData.summary} />
                {battleData.results.map((item) => (
                  <ModelBattleCard key={item.comparison.model_name} item={item} />
                ))}
              </>
            ) : (
              <div className="rounded-3xl border border-dashed border-white/15 bg-white/5 p-10 text-center shadow-xl backdrop-blur">
                <h2 className="text-2xl font-semibold text-white">Ready for battle</h2>
                <p className="mt-3 text-slate-300">
                  Upload a food image and the frontend will call your unified <code>/battle</code> endpoint.
                </p>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}