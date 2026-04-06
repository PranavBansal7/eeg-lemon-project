"use client";

import dynamic from "next/dynamic";

// Keep the route shell thin: the client component owns form state, file uploads,
// and fetch-based demo workflows around the saved benchmark artifacts.
const HomeClient = dynamic(() => import("./HomeClient"), {
  ssr: false,
  loading: function LoadingFallback() {
    return (
      <main className="page-shell">
        <section className="card hero-card">
          <p className="eyebrow">NeuroScope</p>
          <h1>Loading NeuroScope UI...</h1>
          <p className="subtitle">
            Preparing the lightweight demo interface around saved benchmark artifacts.
          </p>
        </section>
      </main>
    );
  },
});

export default function Page() {
  return <HomeClient />;
}
