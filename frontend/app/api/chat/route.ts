import { NextResponse } from "next/server";

import type { FastApiQueryResponse, SourceInfo } from "@/types/chat";

export const runtime = "nodejs";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSourceInfo(value: unknown): value is SourceInfo {
  return (
    isRecord(value) &&
    typeof value.source === "string" &&
    typeof value.page === "number" &&
    typeof value.text === "string" &&
    typeof value.distance === "number"
  );
}

function isFastApiQueryResponse(value: unknown): value is FastApiQueryResponse {
  return (
    isRecord(value) &&
    typeof value.answer === "string" &&
    Array.isArray(value.sources) &&
    value.sources.every(isSourceInfo)
  );
}

export async function POST(request: Request) {
  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ detail: "Request body must be valid JSON." }, { status: 400 });
  }

  const question = isRecord(body) && typeof body.question === "string"
    ? body.question.trim()
    : "";

  if (!question) {
    return NextResponse.json({ detail: "Question is required." }, { status: 400 });
  }

  const fastApiUrl = process.env.FASTAPI_URL?.trim().replace(/\/+$/, "");
  if (!fastApiUrl) {
    return NextResponse.json(
      { detail: "The frontend server is missing FASTAPI_URL." },
      { status: 500 },
    );
  }

  try {
    const upstreamResponse = await fetch(`${fastApiUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      cache: "no-store",
    });

    const responseText = await upstreamResponse.text();
    let payload: unknown = null;

    if (responseText) {
      try {
        payload = JSON.parse(responseText) as unknown;
      } catch {
        payload = null;
      }
    }

    if (!upstreamResponse.ok) {
      const errorPayload = isRecord(payload)
        ? payload
        : { detail: `FastAPI returned HTTP ${upstreamResponse.status}.` };

      return NextResponse.json(errorPayload, { status: upstreamResponse.status });
    }

    if (!isFastApiQueryResponse(payload)) {
      return NextResponse.json(
        { detail: "FastAPI returned an unexpected response." },
        { status: 500 },
      );
    }

    return NextResponse.json(payload);
  } catch (error) {
    console.error("Unable to reach FastAPI /query:", error);
    return NextResponse.json(
      { detail: "Unable to reach the FastAPI service." },
      { status: 500 },
    );
  }
}
