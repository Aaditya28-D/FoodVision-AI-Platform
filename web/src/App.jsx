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

function formatSimilarity(value) {
  if (value == null) return "-";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function prettifyLabel(value) {
  if (!value) return "-";
  return value.replaceAll("_", " ");
}

function ConfidenceBadge({ label }) {
  const styles = {
    "Extremely sure": "border-emerald-200 bg-emerald-50 text-emerald-700",
    "Very confident": "border-cyan-200 bg-cyan-50 text-cyan-700",
    "Fairly confident": "border-amber-200 bg-amber-50 text-amber-700",
    "Somewhat unsure": "border-orange-200 bg-orange-50 text-orange-700",
    "Low confidence": "border-red-200 bg-red-50 text-red-700",
  };

  return (
    <span
      className={`inline-flex rounded-full border px-3 py-1 text-sm font-medium ${
        styles[label] || "border-slate-200 bg-slate-100 text-slate-700"
      }`}
    >
      {label}
    </span>
  );
}

function PageCard({ children, className = "" }) {
  return (
    <div className={`rounded-2xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {children}
    </div>
  );
}

function SectionHeader({ eyebrow, title, description }) {
  return (
    <div className="mb-4">
      {eyebrow ? (
        <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.2em] text-red-600">
          {eyebrow}
        </p>
      ) : null}
      <h3 className="text-xl font-semibold text-slate-900">{title}</h3>
      {description ? <p className="mt-1 text-sm leading-6 text-slate-600">{description}</p> : null}
    </div>
  );
}

function MetricCard({ label, value, tone = "text-slate-900", compact = false }) {
  return (
    <div className={`rounded-xl border border-slate-200 bg-slate-50 ${compact ? "p-3" : "p-4"}`}>
      <p className="text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-1.5 break-words font-semibold ${compact ? "text-lg" : "text-xl"} ${tone}`}>
        {value}
      </p>
    </div>
  );
}

function ChipList({ items, tone = "text-cyan-700", bg = "bg-cyan-50", border = "border-cyan-200" }) {
  if (!items?.length) return <p className="text-sm text-slate-500">Not available.</p>;

  return (
    <div className="flex flex-wrap gap-2.5">
      {items.map((item, index) => (
        <span
          key={`${item}-${index}`}
          className={`rounded-full border px-3 py-1.5 text-sm font-medium ${tone} ${bg} ${border}`}
        >
          {item}
        </span>
      ))}
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-xl px-4 py-2.5 text-sm font-medium transition ${
        active
          ? "bg-slate-900 text-white shadow-sm"
          : "bg-white text-slate-700 border border-slate-200 hover:bg-slate-50"
      }`}
    >
      {children}
    </button>
  );
}

function PredictionList({ predictions }) {
  return (
    <div className="space-y-2">
      {predictions.map((item, index) => (
        <div
          key={`${item.class_name}-${index}`}
          className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5"
        >
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-200 text-[11px] font-semibold text-slate-700">
              {index + 1}
            </span>
            <span className="truncate text-sm font-medium text-slate-800">
              {prettifyLabel(item.class_name)}
            </span>
          </div>
          <span className="ml-3 shrink-0 text-sm font-semibold text-cyan-700">
            {formatConfidence(item.confidence)}
          </span>
        </div>
      ))}
    </div>
  );
}

function BattleSummaryCard({ summary }) {
  if (!summary) return null;

  return (
    <PageCard className="p-4">
      <SectionHeader eyebrow="Model comparison" title="Battle Summary" />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Fastest model" value={summary.fastest_model} tone="text-cyan-700" compact />
        <MetricCard label="Highest confidence" value={summary.highest_confidence_model} tone="text-emerald-700" compact />
        <MetricCard label="Agreement" value={summary.all_models_agree ? "All agree" : "Mixed"} tone="text-amber-700" compact />
        <MetricCard label="Majority label" value={prettifyLabel(summary.majority_label)} tone="text-fuchsia-700" compact />
      </div>
    </PageCard>
  );
}

function OverviewTab({ data, previewUrl }) {
  return (
    <div className="space-y-5">
      <PageCard className="overflow-hidden p-4">
        <SectionHeader
          eyebrow="Main result"
          title={data.food_profile?.food_name || prettifyLabel(data.predicted_class)}
          description="Primary food-level interpretation of the uploaded image."
        />

        <div className="grid gap-5 xl:grid-cols-[280px_1fr]">
          <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
            {previewUrl ? (
              <img src={previewUrl} alt="Uploaded food preview" className="h-72 w-full object-cover" />
            ) : (
              <div className="flex h-72 items-center justify-center text-sm text-slate-500">No image preview</div>
            )}
          </div>

          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2.5">
              <ConfidenceBadge label={data.confidence_label} />
              <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1 text-sm text-slate-700">
                Confidence: {formatConfidence(data.confidence)}
              </span>
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm leading-7 text-slate-700">{data.short_summary}</p>
            </div>

            {data.food_profile ? (
              <div className="grid gap-3 md:grid-cols-3">
                <MetricCard label="Category" value={data.food_profile.category} compact />
                <MetricCard label="Cuisine" value={data.food_profile.cuisine} compact />
                <MetricCard label="Typical serving" value={data.food_profile.serving_info?.typical_serving || "-"} compact />
              </div>
            ) : null}
          </div>
        </div>
      </PageCard>

      <BattleSummaryCard summary={data.battle?.summary} />
    </div>
  );
}

function FoodDetailsTab({ profile }) {
  if (!profile) {
    return (
      <PageCard className="p-4">
        <SectionHeader eyebrow="Food profile" title="Food Details" />
        <p className="text-sm text-slate-600">Detailed food profile is not available yet for this class.</p>
      </PageCard>
    );
  }

  return (
    <div className="space-y-5">
      <PageCard className="p-4">
        <SectionHeader eyebrow="Food profile" title="Basic Information" />
        <div className="grid gap-4 xl:grid-cols-[280px_1fr]">
          <div className="space-y-3">
            <MetricCard label="Food Name" value={profile.food_name} compact />
            <MetricCard label="Category" value={profile.category} compact />
            <MetricCard label="Cuisine" value={profile.cuisine} compact />
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-500">Description</p>
            <p className="mt-2 text-sm leading-7 text-slate-700">{profile.description}</p>
          </div>
        </div>
      </PageCard>

      <div className="grid gap-5 xl:grid-cols-2">
        <PageCard className="p-4">
          <SectionHeader eyebrow="Composition" title="Ingredients" />
          <div className="space-y-5">
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Core ingredients</p>
              <ChipList items={profile.core_ingredients} />
            </div>
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Optional ingredients</p>
              <ChipList
                items={profile.optional_ingredients}
                tone="text-emerald-700"
                bg="bg-emerald-50"
                border="border-emerald-200"
              />
            </div>
          </div>
        </PageCard>

        <PageCard className="p-4">
          <SectionHeader eyebrow="How it is usually made" title="Preparation" />
          <div className="space-y-2.5">
            {profile.preparation_steps.map((step, index) => (
              <div key={`${step}-${index}`} className="flex gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-cyan-100 text-xs font-semibold text-cyan-700">
                  {index + 1}
                </div>
                <p className="text-sm leading-6 text-slate-700">{step}</p>
              </div>
            ))}
          </div>
        </PageCard>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <PageCard className="p-4">
          <SectionHeader eyebrow="Nutrition" title="Nutrition Information" />
          <div className="grid gap-3 md:grid-cols-2">
            <MetricCard label="Calories" value={profile.nutrition.calories} compact />
            <MetricCard label="Protein" value={profile.nutrition.protein} compact />
            <MetricCard label="Carbohydrates" value={profile.nutrition.carbohydrates} compact />
            <MetricCard label="Fat" value={profile.nutrition.fat} compact />
          </div>
        </PageCard>

        <PageCard className="p-4">
          <SectionHeader eyebrow="Health view" title="Health Advice" />
          <div className="space-y-2.5">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-500">Benefits</p>
              <p className="mt-1.5 text-sm leading-6 text-slate-700">{profile.health_advice.benefits}</p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-500">Risks</p>
              <p className="mt-1.5 text-sm leading-6 text-slate-700">{profile.health_advice.risks}</p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-xs uppercase tracking-wide text-slate-500">Healthy tip</p>
              <p className="mt-1.5 text-sm leading-6 text-slate-700">{profile.health_advice.healthy_tip}</p>
            </div>
          </div>
        </PageCard>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <PageCard className="p-4">
          <SectionHeader eyebrow="Dietary view" title="Dietary Notes" />
          <div className="space-y-4">
            <ChipList
              items={[
                profile.dietary_notes.vegetarian_possible ? "Vegetarian possible" : null,
                profile.dietary_notes.non_vegetarian_possible ? "Non-vegetarian possible" : null,
                profile.dietary_notes.vegan_possible ? "Vegan possible" : null,
                profile.dietary_notes.gluten_free_possible ? "Gluten-free possible" : null,
              ].filter(Boolean)}
              tone="text-amber-700"
              bg="bg-amber-50"
              border="border-amber-200"
            />
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <p className="text-sm leading-6 text-slate-700">{profile.dietary_notes.note}</p>
            </div>
          </div>
        </PageCard>

        <PageCard className="p-4">
          <SectionHeader eyebrow="Safety" title="Allergens" />
          <ChipList
            items={profile.allergens}
            tone="text-red-700"
            bg="bg-red-50"
            border="border-red-200"
          />
        </PageCard>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <PageCard className="p-4">
          <SectionHeader eyebrow="Common forms" title="Variations" />
          <ChipList
            items={profile.common_variations}
            tone="text-fuchsia-700"
            bg="bg-fuchsia-50"
            border="border-fuchsia-200"
          />
        </PageCard>

        <PageCard className="p-4">
          <SectionHeader eyebrow="How it is served" title="Serving and Regions" />
          <div className="space-y-4">
            <MetricCard label="Typical serving" value={profile.serving_info.typical_serving} compact />
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Best served with</p>
              <ChipList
                items={profile.serving_info.best_served_with}
                tone="text-cyan-700"
                bg="bg-cyan-50"
                border="border-cyan-200"
              />
            </div>
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Popular regions</p>
              <ChipList
                items={profile.popular_regions}
                tone="text-emerald-700"
                bg="bg-emerald-50"
                border="border-emerald-200"
              />
            </div>
          </div>
        </PageCard>
      </div>
    </div>
  );
}

function ExplainabilityTab({ battle }) {
  if (!battle?.results?.length) return null;

  return (
    <div className="space-y-5">
      <BattleSummaryCard summary={battle.summary} />

      {battle.results.map((item) => {
        const topPrediction = item.comparison.top_prediction;
        const heatmapUrl = `${API_BASE_URL}${item.explanation.heatmap_url}`;

        return (
          <PageCard key={item.comparison.model_name} className="p-4">
            <SectionHeader
              eyebrow="Model reasoning"
              title={item.comparison.model_name}
              description="Confidence, top predictions, and Grad-CAM attention map."
            />

            <div className="grid gap-5 xl:grid-cols-[1fr_420px]">
              <div className="space-y-4 min-w-0">
                <div className="grid gap-3 md:grid-cols-3">
                  <MetricCard label="Top prediction" value={prettifyLabel(topPrediction.class_name)} tone="text-emerald-700" compact />
                  <MetricCard label="Confidence" value={formatConfidence(topPrediction.confidence)} tone="text-cyan-700" compact />
                  <MetricCard label="Speed" value={formatMs(item.comparison.inference_time_ms)} compact />
                </div>

                <div>
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Top-{item.comparison.predictions.length} predictions
                  </p>
                  <PredictionList predictions={item.comparison.predictions} />
                </div>
              </div>

              <div>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  Grad-CAM Explanation
                </p>
                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
                  <img
                    src={heatmapUrl}
                    alt={`${item.comparison.model_name} Grad-CAM`}
                    className="h-auto w-full object-contain"
                  />
                </div>
              </div>
            </div>
          </PageCard>
        );
      })}
    </div>
  );
}

function RetrievalImageCard({ item, isSameClass = false }) {
  const imageUrl = `${API_BASE_URL}${item.image_url}`;

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
      <a href={imageUrl} target="_blank" rel="noreferrer" className="block overflow-hidden">
        <img
          src={imageUrl}
          alt={item.class_name}
          className="h-52 w-full object-cover transition hover:scale-[1.02]"
        />
      </a>

      <div className="space-y-3 p-4">
        <div className="flex items-center justify-between gap-3">
          <span className="rounded-full bg-slate-900 px-2.5 py-1 text-xs font-medium text-white">
            #{item.rank}
          </span>
          <span className="text-sm font-semibold text-cyan-700">
            {formatSimilarity(item.similarity)}
          </span>
        </div>

        {isSameClass ? (
          <div>
            <span className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
              Matches predicted class
            </span>
          </div>
        ) : null}

        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Matched class</p>
          <p className="mt-1 text-lg font-semibold text-slate-900">
            {prettifyLabel(item.class_name)}
          </p>
        </div>

        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">Dataset image</p>
          <p className="mt-1 break-all text-sm text-slate-600">
            {item.image_path}
          </p>
        </div>
      </div>
    </div>
  );
}

function SimilarSummaryCard({ retrieval }) {
  const sameCount = retrieval?.same_class_results?.length || 0;
  const otherCount = retrieval?.other_results?.length || 0;

  return (
    <PageCard className="p-4">
      <SectionHeader
        eyebrow="Retrieval summary"
        title="Similarity Overview"
        description="Quick summary of exact match status and retrieval grouping."
      />
      <div className="grid gap-3 md:grid-cols-4">
        <MetricCard label="Predicted class" value={prettifyLabel(retrieval?.predicted_class)} tone="text-cyan-700" compact />
        <MetricCard label="Exact match" value={retrieval?.exact_match_found ? "Found" : "Not found"} tone={retrieval?.exact_match_found ? "text-emerald-700" : "text-slate-900"} compact />
        <MetricCard label="Same-class results" value={String(sameCount)} tone="text-emerald-700" compact />
        <MetricCard label="Other visual matches" value={String(otherCount)} tone="text-amber-700" compact />
      </div>
    </PageCard>
  );
}

function SimilarDishesTab({ retrieval }) {
  const [showAllSame, setShowAllSame] = useState(false);
  const [showAllOther, setShowAllOther] = useState(false);

  if (!retrieval) {
    return (
      <PageCard className="p-4">
        <SectionHeader eyebrow="Retrieval" title="Similar Dishes" description="No similar dish results are available yet." />
      </PageCard>
    );
  }

  const visibleSame = showAllSame
    ? retrieval.same_class_results || []
    : (retrieval.same_class_results || []).slice(0, 3);

  const visibleOther = showAllOther
    ? retrieval.other_results || []
    : (retrieval.other_results || []).slice(0, 3);

  return (
    <div className="space-y-5">
      <SimilarSummaryCard retrieval={retrieval} />

      {retrieval.exact_match_found ? (
        <PageCard className="p-4">
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
            An exact or near-exact dataset match was found for this uploaded image.
          </div>
        </PageCard>
      ) : null}

      <PageCard className="p-4">
        <SectionHeader
          eyebrow="Same predicted food"
          title="Same-Class Similar Dishes"
          description="These matches belong to the same predicted class and are the most useful examples of similar dishes."
        />

        {visibleSame.length > 0 ? (
          <>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {visibleSame.map((item) => (
                <RetrievalImageCard
                  key={`same-${item.rank}-${item.image_path}`}
                  item={item}
                  isSameClass
                />
              ))}
            </div>

            {(retrieval.same_class_results || []).length > 3 ? (
              <div className="mt-5 flex justify-center">
                <button
                  type="button"
                  onClick={() => setShowAllSame((prev) => !prev)}
                  className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                >
                  {showAllSame
                    ? "Show fewer same-class matches"
                    : `Show all ${(retrieval.same_class_results || []).length} same-class matches`}
                </button>
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-slate-600">No same-class retrieval results were found.</p>
        )}
      </PageCard>

      <PageCard className="p-4">
        <SectionHeader
          eyebrow="Other visual matches"
          title="Related Looking Dishes"
          description="These are dishes from other classes that look visually similar in embedding space."
        />

        {visibleOther.length > 0 ? (
          <>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {visibleOther.map((item) => (
                <RetrievalImageCard
                  key={`other-${item.rank}-${item.image_path}`}
                  item={item}
                />
              ))}
            </div>

            {(retrieval.other_results || []).length > 3 ? (
              <div className="mt-5 flex justify-center">
                <button
                  type="button"
                  onClick={() => setShowAllOther((prev) => !prev)}
                  className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                >
                  {showAllOther
                    ? "Show fewer other matches"
                    : `Show all ${(retrieval.other_results || []).length} other matches`}
                </button>
              </div>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-slate-600">No other-class visual matches were found.</p>
        )}
      </PageCard>
    </div>
  );
}

function LoadingState() {
  return (
    <PageCard className="p-8 text-center">
      <div className="mx-auto h-12 w-12 animate-spin rounded-full border-4 border-cyan-200 border-t-cyan-500" />
      <h2 className="mt-4 text-xl font-semibold text-slate-900">Analyzing image</h2>
      <p className="mt-2 text-sm text-slate-600">
        Running model comparison, generating food details, preparing explainability, and retrieving similar dishes.
      </p>
    </PageCard>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [analyzeData, setAnalyzeData] = useState(null);
  const [retrievalData, setRetrievalData] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");

  const preview = useMemo(() => previewUrl, [previewUrl]);

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setAnalyzeData(null);
    setRetrievalData(null);
    setError("");
    setActiveTab("overview");

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
      setAnalyzeData(null);
      setRetrievalData(null);

      const analyzeFormData = new FormData();
      analyzeFormData.append("image", selectedFile);

      const analyzeResponse = await fetch(`${API_BASE_URL}/analyze?top_k=${topK}`, {
        method: "POST",
        body: analyzeFormData,
      });

      const analyzeJson = await analyzeResponse.json();

      if (!analyzeResponse.ok) {
        throw new Error(analyzeJson.detail || "Analyze request failed.");
      }

      const retrievalFormData = new FormData();
      retrievalFormData.append("image", selectedFile);

      const retrievalResponse = await fetch(`${API_BASE_URL}/retrieve/similar?top_k=6`, {
        method: "POST",
        body: retrievalFormData,
      });

      const retrievalJson = await retrievalResponse.json();

      if (!retrievalResponse.ok) {
        throw new Error(retrievalJson.detail || "Retrieval request failed.");
      }

      setAnalyzeData(analyzeJson);
      setRetrievalData(retrievalJson);
      setActiveTab("overview");
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#fcfbf7] px-4 py-6 text-slate-900 md:px-6 xl:px-8">
      <div className="mx-auto max-w-7xl">
        <header className="mb-6 overflow-hidden rounded-[28px] border border-[#12233a] bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.18),_transparent_24%),linear-gradient(135deg,#081425_0%,#0d2240_100%)] px-6 py-8 shadow-xl">
          <div className="inline-flex items-center rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium tracking-wide text-cyan-300">
            FoodVision AI Platform
          </div>
          <h1 className="mt-3 text-3xl font-bold tracking-tight text-white md:text-4xl font-serif">
            Food Analysis Dashboard
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-7 text-slate-200 md:text-base">
            Upload a food image to identify the main food, understand confidence, explore food information,
            inspect explainability, and retrieve visually similar dishes.
          </p>
        </header>

        <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
          <aside className="xl:sticky xl:top-6 xl:self-start">
            <PageCard className="p-4">
              <h2 className="text-lg font-semibold text-slate-900 font-serif">Upload Image</h2>

              <form onSubmit={handleAnalyze} className="mt-4 space-y-4">
                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-700">Food image</label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="block w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-700 file:mr-3 file:rounded-lg file:border-0 file:bg-cyan-500 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-cyan-400"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-700">Top-K predictions</label>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-sm text-slate-700 outline-none focus:border-cyan-400"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full rounded-xl bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Analyzing..." : "Analyze Food"}
                </button>
              </form>

              {error ? (
                <div className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                  {error}
                </div>
              ) : null}

              <div className="mt-5">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Preview</h3>
                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-slate-50">
                  {preview ? (
                    <img src={preview} alt="Selected upload preview" className="h-64 w-full object-cover" />
                  ) : (
                    <div className="flex h-64 items-center justify-center px-6 text-center text-sm text-slate-500">
                      Choose an image to preview it here.
                    </div>
                  )}
                </div>
              </div>
            </PageCard>
          </aside>

          <main className="space-y-5 min-w-0">
            {loading ? <LoadingState /> : null}

            {!loading && analyzeData ? (
              <>
                <PageCard className="p-3">
                  <div className="flex flex-wrap gap-2.5">
                    <TabButton active={activeTab === "overview"} onClick={() => setActiveTab("overview")}>
                      Overview
                    </TabButton>
                    <TabButton active={activeTab === "details"} onClick={() => setActiveTab("details")}>
                      Food Details
                    </TabButton>
                    <TabButton active={activeTab === "explainability"} onClick={() => setActiveTab("explainability")}>
                      Explainability
                    </TabButton>
                    <TabButton active={activeTab === "similar"} onClick={() => setActiveTab("similar")}>
                      Similar Dishes
                    </TabButton>
                  </div>
                </PageCard>

                {activeTab === "overview" ? <OverviewTab data={analyzeData} previewUrl={preview} /> : null}
                {activeTab === "details" ? <FoodDetailsTab profile={analyzeData.food_profile} /> : null}
                {activeTab === "explainability" ? <ExplainabilityTab battle={analyzeData.battle} /> : null}
                {activeTab === "similar" ? <SimilarDishesTab retrieval={retrievalData} /> : null}
              </>
            ) : null}

            {!loading && !analyzeData ? (
              <PageCard className="p-8 text-center">
                <h2 className="text-xl font-semibold text-slate-900 font-serif">Ready to analyze</h2>
                <p className="mt-2 text-sm text-slate-600">
                  Upload a food image to generate a prediction, food details, explainability, and similar-dish retrieval.
                </p>
              </PageCard>
            ) : null}
          </main>
        </div>
      </div>
    </div>
  );
}