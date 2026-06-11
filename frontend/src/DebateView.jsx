import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";

const API_BASE = "http://localhost:8000";

const AGENT_POS = {
  agent1: { x: 200, y: 80 },
  agent2: { x: 80, y: 280 },
  agent3: { x: 320, y: 280 },
};
const AGENT_COLORS = {
  agent1: "#1a1917",
  agent2: "#4a6fa5",
  agent3: "#7a4f9a",
};
const AGENT_NAMES = {
  agent1: "Semantic Agent",
  agent2: "Ego Agent",
  agent3: "Observer Agent",
};

function AgentNode({ id, pos, color, answer, active }) {
  return (
    <g>
      <circle
        cx={pos.x}
        cy={pos.y}
        r={36}
        fill={active ? color : "#f5f4f0"}
        stroke={color}
        strokeWidth={2}
        style={{ transition: "fill 0.4s" }}
      />
      <text x={pos.x} y={pos.y - 6} textAnchor="middle" fill={active ? "#fff" : color} fontSize={9} fontWeight={600}>
        {AGENT_NAMES[id] || id}
      </text>
      <text x={pos.x} y={pos.y + 12} textAnchor="middle" fill={active ? "#fff" : "#888780"} fontSize={13} fontWeight={700}>
        {answer || "—"}
      </text>
    </g>
  );
}

function Arrow({ from, to, color, label, visible }) {
  if (!visible) return null;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const r = 38;
  const sx = from.x + (dx / len) * r;
  const sy = from.y + (dy / len) * r;
  const ex = to.x - (dx / len) * r;
  const ey = to.y - (dy / len) * r;
  const mx = (sx + ex) / 2;
  const my = (sy + ey) / 2;

  return (
    <g>
      <defs>
        <marker id={`arrow-${color.replace("#","")}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill={color} />
        </marker>
      </defs>
      <line
        x1={sx} y1={sy} x2={ex} y2={ey}
        stroke={color} strokeWidth={1.5}
        markerEnd={`url(#arrow-${color.replace("#","")})`}
        strokeDasharray="4 3"
        style={{ transition: "opacity 0.3s" }}
      />
      {label && (
        <text x={mx} y={my - 6} textAnchor="middle" fill={color} fontSize={10}>
          {label}
        </text>
      )}
    </g>
  );
}

export default function DebateView() {
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get("session");

  const [phase, setPhase] = useState("waiting"); // waiting | agent_answer | critique | rebuttal | consensus | done | error
  const [round, setRound] = useState(0);
  const [answers, setAnswers] = useState({});
  const [arrows, setArrows] = useState([]); // [{from, to, color, label}]
  const [logs, setLogs] = useState([]);
  const [connected, setConnected] = useState(false);
  const logRef = useRef(null);

  const addLog = (type, text) => {
    setLogs((prev) => [...prev, { type, text, ts: Date.now() }]);
  };

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    if (!sessionId) return;

    const es = new EventSource(`${API_BASE}/debate-stream/${sessionId}`);
    setConnected(true);

    es.addEventListener("agent_answer", (e) => {
      const data = JSON.parse(e.data);
      const newAnswers = {};
      for (const [ak, ans] of Object.entries(data.answers)) {
        newAnswers[ak] = ans[0]?.value || "?";
      }
      setAnswers(newAnswers);
      setPhase("agent_answer");
      setArrows([]);
      addLog("agent_answer", `초기 추론 완료 — ${Object.entries(newAnswers).map(([k,v]) => `${AGENT_NAMES[k] || k}: ${v}`).join(", ")}`);
    });

    es.addEventListener("critique", (e) => {
      const data = JSON.parse(e.data);
      setRound(data.round);
      setPhase("critique");

      const newArrows = [];
      for (const [src, targets] of Object.entries(data.critiques)) {
        for (const [tgt, text] of Object.entries(targets)) {
          if (text) {
            newArrows.push({ from: src, to: tgt, color: "#e07b3a", label: "critique" });
            addLog("critique", `[R${data.round}] ${AGENT_NAMES[src] || src} → ${AGENT_NAMES[tgt] || tgt}: ${text.slice(0, 120)}${text.length > 120 ? "..." : ""}`);
          }
        }
      }
      setArrows(newArrows);
    });

    es.addEventListener("rebuttal", (e) => {
      const data = JSON.parse(e.data);
      setPhase("rebuttal");
      setRound(data.round);

      const newAnswers = {};
      for (const [ak, ans] of Object.entries(data.answers_after)) {
        newAnswers[ak] = ans[0]?.value || "?";
      }
      setAnswers(newAnswers);

      const allAgentIds = ["agent1", "agent2", "agent3"];
      const newArrows = [];
      for (const [ak, text] of Object.entries(data.rebuttals)) {
        if (text) {
          for (const other of allAgentIds) {
            if (other !== ak) {
              newArrows.push({ from: ak, to: other, color: "#4a6fa5", label: "rebuttal" });
            }
          }
          addLog("rebuttal", `[R${data.round}] ${AGENT_NAMES[ak] || ak} rebuttal: ${text.slice(0, 120)}${text.length > 120 ? "..." : ""}`);
        }
      }
      setArrows(newArrows);
    });

    es.addEventListener("consensus", (e) => {
      const data = JSON.parse(e.data);
      setPhase("consensus");
      setArrows([]);
      const mapStr = Object.entries(data.answer_map || {})
        .map(([qid, votes]) =>
          `${qid}: ${Object.entries(votes).map(([a, v]) => `${a}=${v}`).join(", ")}`
        )
        .join(" | ");
      if (data.agreement) {
        addLog("consensus", `[R${data.round}] ✓ 합의 도달${mapStr ? ` — ${mapStr}` : ""}`);
      } else {
        addLog("consensus", `[R${data.round}] ✗ 합의 실패${mapStr ? ` — ${mapStr}` : ""} — 다음 라운드 또는 majority vote`);
      }
    });

    es.addEventListener("done", (e) => {
      setPhase("done");
      setConnected(false);
      addLog("done", "파이프라인 완료");
      es.close();
    });

    es.addEventListener("error", (e) => {
      setPhase("error");
      setConnected(false);
      const msg = e.data ? JSON.parse(e.data).message : "연결 오류";
      addLog("error", msg);
      es.close();
    });

    return () => es.close();
  }, [sessionId]);

  const agentIds = ["agent1", "agent2", "agent3"];

  return (
    <div style={s.root}>
      <div style={s.header}>
        <span style={s.logo}>PRISM</span>
        <span style={s.subtitle}>Debate Visualizer</span>
        <span style={{ ...s.badge, background: connected ? "#d4edda" : "#f1efe8", color: connected ? "#2d6a4f" : "#888780" }}>
          {connected ? "● live" : phase === "done" ? "완료" : "대기 중"}
        </span>
        {round > 0 && <span style={s.badge}>Round {round}</span>}
      </div>

      <div style={s.body}>
        {/* SVG graph */}
        <div style={s.graphPanel}>
          <svg viewBox="0 0 400 380" width="100%" height="100%">
            {/* arrows */}
            {arrows.filter(a => a.from !== a.to).map((a, i) => (
              <Arrow
                key={i}
                from={AGENT_POS[a.from]}
                to={AGENT_POS[a.to]}
                color={a.color}
                label={a.label}
                visible={true}
              />
            ))}
            {/* nodes */}
            {agentIds.map((id) => (
              <AgentNode
                key={id}
                id={id}
                pos={AGENT_POS[id]}
                color={AGENT_COLORS[id]}
                answer={answers[id]}
                active={phase !== "waiting"}
              />
            ))}
            {/* phase label */}
            <text x={200} y={360} textAnchor="middle" fill="#888780" fontSize={12}>
              {phase === "waiting" && "연결 대기 중..."}
              {phase === "agent_answer" && "초기 추론"}
              {phase === "critique" && `Round ${round} — Critique`}
              {phase === "rebuttal" && `Round ${round} — Rebuttal`}
              {phase === "consensus" && `Round ${round} — Consensus Check`}
              {phase === "done" && "완료"}
              {phase === "error" && "오류"}
            </text>
          </svg>
        </div>

        {/* log panel */}
        <div style={s.logPanel} ref={logRef}>
          {logs.length === 0 && (
            <div style={s.logEmpty}>이벤트를 기다리는 중...</div>
          )}
          {logs.map((log, i) => (
            <div key={i} style={{ ...s.logItem, ...logTypeStyle(log.type) }}>
              <span style={s.logTag}>{log.type}</span>
              <span style={s.logText}>{log.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function logTypeStyle(type) {
  switch (type) {
    case "critique": return { borderLeft: "3px solid #e07b3a" };
    case "rebuttal": return { borderLeft: "3px solid #4a6fa5" };
    case "consensus": return { borderLeft: "3px solid #2d6a4f" };
    case "agent_answer": return { borderLeft: "3px solid #1a1917" };
    case "done": return { borderLeft: "3px solid #888780" };
    case "error": return { borderLeft: "3px solid #a32d2d", background: "#fcebeb" };
    default: return {};
  }
}

const s = {
  root: {
    height: "100vh",        // minHeight → height 로 변경
    overflow: "hidden",     // 추가
    background: "#f5f4f0",
    fontFamily: "'IBM Plex Sans', 'Noto Sans KR', sans-serif",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    padding: "16px 28px",
    borderBottom: "0.5px solid #d3d1c7",
    background: "#fff",
  },
  logo: {
    fontSize: 16,
    fontWeight: 700,
    letterSpacing: "0.12em",
    fontFamily: "'IBM Plex Mono', monospace",
    color: "#1a1917",
  },
  subtitle: {
    fontSize: 13,
    color: "#888780",
  },
  badge: {
    fontSize: 11,
    padding: "3px 8px",
    borderRadius: 6,
    background: "#f1efe8",
    color: "#5f5e5a",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  body: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
    height: "calc(100vh - 57px)",
    minHeight: 0,
  },
  graphPanel: {
    flex: 0.8,
    borderRight: "0.5px solid #d3d1c7",
    background: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: 16,
    height: "100%",
  },
  logPanel: {
    flex: 1,
    overflowY: "auto",
    height: "100%",
    minHeight: 0,
    padding: "16px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  logEmpty: {
    fontSize: 13,
    color: "#b4b2a9",
    marginTop: 12,
  },
  logItem: {
    background: "#fff",
    border: "0.5px solid #d3d1c7",
    borderRadius: 8,
    padding: "10px 14px",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  logTag: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "#888780",
    fontFamily: "'IBM Plex Mono', monospace",
  },
  logText: {
    fontSize: 13,
    color: "#1a1917",
    lineHeight: 1.5,
    wordBreak: "break-word",
  },
};