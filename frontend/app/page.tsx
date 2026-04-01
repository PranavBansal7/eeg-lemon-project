"use client";

import dynamic from "next/dynamic";

const HomeClient = dynamic(() => import("./HomeClient"), {
  ssr: false,
  loading: function LoadingFallback() {
    return (
      <main className="page-shell">
        <section className="card hero-card">
          <p className="eyebrow">NeuroScope</p>
          <h1>Loading NeuroScope UI...</h1>
          <p className="subtitle">Preparing the prediction interface.</p>
        </section>
      </main>
    );
  },
});

export default function Page() {
  return <HomeClient />;
}
