export interface SourceInfo {
  source: string;
  page: number;
  text: string;
  distance: number;
}

export interface FastApiQueryResponse {
  answer: string;
  sources: SourceInfo[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceInfo[];
}
