"use client";

import dynamic from "next/dynamic";

const RetrievalApp = dynamic(
  () => import("../../components/RetrievalApp").then((m) => m.RetrievalApp),
  { ssr: false },
);

export default function RetrievalPage() {
  return <RetrievalApp />;
}

