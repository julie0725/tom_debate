import { useState, useRef } from "react";
import { Routes, Route } from 'react-router-dom'
import DebateView from './DebateView.jsx'

const API_URL = "http://localhost:8000/run";

const STEPS = [
  { key: "pipeline_start", label: "파이프라인 시작", pct: 5 },
  { key: "agents_running", label: "에이전트 추론 중", pct: 30 },
  { key: "debate_start", label: "토론 시작", pct: 50 },
  { key: "debate_done", label: "토론 완료", pct: 80 },
  { key: "done", label: "완료", pct: 100 },
];

export default function App() {
  const [scenario, setScenario] = useState("");
  const [question, setQuestion] = useState("");
  const [question2, setQuestion2] = useState("");
  const [question3, setQuestion3] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ pct: 0, message: "" });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const handleRun = async () => {
    if (!scenario.trim() || !question.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setProgress({ pct: 0, message: "" });

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: scenario.trim(), question: question.trim(), question2: question2.trim(), question3: question3.trim() }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      abortRef.current = reader;

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          if (!part.trim()) continue;
          const lines = part.split("\n");
          let eventType = "message";
          let dataStr = "";
          for (const line of lines) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          }
          if (!dataStr) continue;
          try {
            const data = JSON.parse(dataStr);
            if (eventType === "progress") {
              setProgress({ pct: data.pct, message: data.message });
            } else if (eventType === "session") {
              window.open(`/debate?session=${data.session_id}`, "_blank");
            } else if (eventType === "done") {
              setProgress({ pct: 100, message: "완료" });
              setResult(data);
              setLoading(false);
            } else if (eventType === "error") {
              setError(data.message);
              setLoading(false);
            }
          } catch {}
        }
      }
    } catch (e) {
      setError(e.message);
      setLoading(false);
    }
  };

  const handleReset = () => {
    setScenario("");
    setQuestion("");
    setQuestion2("");
    setQuestion3("");
    setResult(null);
    setError(null);
    setProgress({ pct: 0, message: "" });
  };

  const mainPage = (
    <div style={styles.root}>
      <div style={styles.container}>
        <header style={styles.header}>
          <div style={styles.logo}>PRISM</div>
          <div style={styles.subtitle}>Theory of Mind Reasoning System</div>
        </header>

        <div style={styles.card}>
          <label style={styles.label}>Scenario</label>
          <textarea
            style={{ ...styles.textarea, height: 160 }}
            placeholder="시나리오를 입력하세요..."
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            disabled={loading}
          />

          <label style={{ ...styles.label, marginTop: 20 }}>Question 1</label>
          <textarea
            style={{ ...styles.textarea, height: 72 }}
            placeholder="질문 1을 입력하세요..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
          />

          <label style={{ ...styles.label, marginTop: 20 }}>Question 2 <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0, color: "#b4b2a9" }}>(선택)</span></label>
          <textarea
            style={{ ...styles.textarea, height: 72 }}
            placeholder="질문 2를 입력하세요 (없으면 비워두세요)..."
            value={question2}
            onChange={(e) => setQuestion2(e.target.value)}
            disabled={loading}
          />

          <label style={{ ...styles.label, marginTop: 20 }}>Question 3 <span style={{ fontWeight: 400, textTransform: "none", letterSpacing: 0, color: "#b4b2a9" }}>(선택)</span></label>
          <textarea
            style={{ ...styles.textarea, height: 72 }}
            placeholder="질문 3을 입력하세요 (없으면 비워두세요)..."
            value={question3}
            onChange={(e) => setQuestion3(e.target.value)}
            disabled={loading}
          />

          <div style={styles.buttonRow}>
            <button
              style={{
                ...styles.btn,
                ...(loading || !scenario.trim() || !question.trim()
                  ? styles.btnDisabled
                  : styles.btnPrimary),
              }}
              onClick={handleRun}
              disabled={loading || !scenario.trim() || !question.trim()}
            >
              {loading ? "실행 중..." : "실행"}
            </button>
            {(result || error) && (
              <button style={{ ...styles.btn, ...styles.btnGhost }} onClick={handleReset}>
                초기화
              </button>
            )}
          </div>
        </div>

        {loading && (
          <div style={styles.card}>
            <div style={styles.progressLabel}>
              <span style={styles.progressMessage}>{progress.message || "준비 중..."}</span>
              <span style={styles.progressPct}>{progress.pct}%</span>
            </div>
            <div style={styles.progressTrack}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${progress.pct}%`,
                  transition: "width 0.4s ease",
                }}
              />
            </div>
          </div>
        )}

        {error && (
          <div style={{ ...styles.card, ...styles.errorCard }}>
            <span style={styles.errorText}>오류: {error}</span>
          </div>
        )}

        {result && (
          <div style={styles.card}>
            <div style={styles.resultHeader}>
              <span style={styles.resultTitle}>결과</span>
              <div style={styles.metaRow}>
                <span style={styles.metaBadge}>status: {result.status ?? "-"}</span>
                <span style={styles.metaBadge}>debate rounds: {result.debate_round ?? 0}</span>
                {result.debate_triggered && (
                  <span style={{ ...styles.metaBadge, ...styles.metaBadgeDebate }}>debate triggered</span>
                )}
              </div>
            </div>
            <div style={styles.answerGrid}>
              {[
                { id: "q1", label: "Q1 · Belief", value: result.q1 },
                { id: "q2", label: "Q2 · Desire", value: result.q2 },
                { id: "q3", label: "Q3 · Action", value: result.q3 },
              ]
                .filter((q) => q.value != null)
                .map((q) => (
                  <div key={q.id} style={styles.answerCard}>
                    <div style={styles.answerLabel}>{q.label}</div>
                    <div style={styles.answerValue}>{q.value}</div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <Routes>
      <Route path="/" element={mainPage} />
      <Route path="/debate" element={<DebateView />} />
    </Routes>
  );
}

const styles = {
  root: {
    minHeight: "100vh",
    background: "#f5f4f0",
    display: "flex",
    justifyContent: "center",
    padding: "48px 24px",
    fontFamily: "'IBM Plex Sans', 'Noto Sans KR', sans-serif",
    boxSizing: "border-box",
    width: "100%",
    overflowX: "hidden",
  },
  container: {
    width: "100%",
    maxWidth: 640,
    minWidth: 0,
  },
  header: {
    marginBottom: 32,
    textAlign: "center",
  },
  logo: {
    fontSize: 28,
    fontWeight: 700,
    letterSpacing: "0.12em",
    color: "#1a1917",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  subtitle: {
    fontSize: 13,
    color: "#888780",
    marginTop: 4,
    letterSpacing: "0.04em",
  },
  card: {
    background: "#ffffff",
    border: "0.5px solid #d3d1c7",
    borderRadius: 12,
    padding: "24px 28px",
    marginBottom: 16,
  },
  label: {
    display: "block",
    fontSize: 12,
    fontWeight: 500,
    letterSpacing: "0.08em",
    color: "#888780",
    textTransform: "uppercase",
    marginBottom: 8,
  },
  textarea: {
    width: "100%",
    boxSizing: "border-box",
    resize: "vertical",
    border: "0.5px solid #d3d1c7",
    borderRadius: 8,
    padding: "12px 14px",
    fontSize: 14,
    lineHeight: 1.6,
    color: "#1a1917",
    background: "#fafaf8",
    outline: "none",
    fontFamily: "inherit",
  },
  buttonRow: {
    display: "flex",
    gap: 10,
    marginTop: 20,
  },
  btn: {
    padding: "10px 24px",
    borderRadius: 8,
    fontSize: 14,
    fontWeight: 500,
    cursor: "pointer",
    border: "0.5px solid #d3d1c7",
    transition: "opacity 0.15s",
  },
  btnPrimary: {
    background: "#1a1917",
    color: "#f5f4f0",
    border: "0.5px solid #1a1917",
  },
  btnDisabled: {
    background: "#d3d1c7",
    color: "#888780",
    cursor: "not-allowed",
  },
  btnGhost: {
    background: "transparent",
    color: "#5f5e5a",
  },
  progressLabel: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 10,
  },
  progressMessage: {
    fontSize: 13,
    color: "#5f5e5a",
  },
  progressPct: {
    fontSize: 13,
    fontWeight: 500,
    color: "#1a1917",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  progressTrack: {
    height: 4,
    background: "#f1efe8",
    borderRadius: 2,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "#1a1917",
    borderRadius: 2,
  },
  errorCard: {
    background: "#fcebeb",
    border: "0.5px solid #f09595",
  },
  errorText: {
    fontSize: 13,
    color: "#a32d2d",
  },
  resultHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 20,
  },
  resultTitle: {
    fontSize: 12,
    fontWeight: 500,
    letterSpacing: "0.08em",
    color: "#888780",
    textTransform: "uppercase",
  },
  metaRow: {
    display: "flex",
    gap: 6,
    flexWrap: "wrap",
    justifyContent: "flex-end",
  },
  metaBadge: {
    fontSize: 11,
    padding: "3px 8px",
    borderRadius: 6,
    background: "#f1efe8",
    color: "#5f5e5a",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  metaBadgeDebate: {
    background: "#faeeda",
    color: "#854f0b",
  },
  answerGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
    gap: 12,
  },
  answerCard: {
    background: "#f5f4f0",
    borderRadius: 8,
    padding: "14px 16px",
  },
  answerLabel: {
    fontSize: 11,
    fontWeight: 500,
    letterSpacing: "0.06em",
    color: "#888780",
    marginBottom: 6,
  },
  answerValue: {
    fontSize: 15,
    fontWeight: 500,
    color: "#1a1917",
    wordBreak: "break-word",
  },
};