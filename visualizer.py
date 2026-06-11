# visualizer.py
# streamlit run visualizer.py로 실행

import streamlit as st
import matplotlib.pyplot as plt
import json
import time
import numpy as np

st.set_page_config(page_title="Debate Visualizer")
st.title("Agent Debate Visualizer")

log_path = st.text_input("log file", value="outputs/debate_round_01.json")
if not log_path:
    st.stop()
with open(log_path) as f:
    data = json.load(f)

round_num   = data["_round"]
ws          = data["what_agents_see"]
before      = data["agent_outputs_before_reInfer"]
after       = data["agent_outputs_after_reInfer"]
supervisor  = data["supervisor_result"]

AGENT_NAMES = [k for k in ws.keys() if k.startswith("agent")]

NODE_COLORS = {"agent1": "#AFA9EC", "agent2": "#5DCAA5", "agent3": "#F0997B"}
POS = {
    "agent1": np.array([0.0,   0.80]),
    "agent2": np.array([-0.70, -0.40]),
    "agent3": np.array([ 0.70, -0.40]),
}

critique_edges = []
for src in AGENT_NAMES:
    for key, text in ws[src]["critiques_given"].items():
        tgt = key.replace("critique_of_", "")
        critique_edges.append((src, tgt, text))

answer_changes = {}
for agent in AGENT_NAMES:
    changes = {}
    for q in before[agent]:
        qid  = q["id"]
        b    = q["value"]
        a_v  = next(x["value"] for x in after[agent] if x["id"] == qid)
        changes[qid] = (b, a_v, b != a_v)
    answer_changes[agent] = changes

def draw(active_nodes, active_edges, highlight_nodes, node_answers, phase_label,
         latest_edge_idx=None):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.05, 1.40)
    ax.axis("off")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for i, (src, tgt, label) in enumerate(active_edges):
        ps, pt = POS[src], POS[tgt]
        d = pt - ps
        dn = d / np.linalg.norm(d)
        s  = ps + 0.22 * dn
        e  = pt - 0.22 * dn
        is_latest = (latest_edge_idx is not None and i == latest_edge_idx)
        color = "#FFD700" if is_latest else "#666666"
        lw    = 2.2 if is_latest else 0.9
        alpha = 1.0 if is_latest else 0.45
        ax.annotate("", xy=e, xytext=s,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                    alpha=alpha)

    for agent in active_nodes:
        x, y = POS[agent]
        clr  = NODE_COLORS[agent]
        ring = "#FFD700" if agent in highlight_nodes else clr
        rlw  = 3.0 if agent in highlight_nodes else 1.2
        ax.add_patch(plt.Circle((x, y), 0.18, color=clr, zorder=5))
        ax.add_patch(plt.Circle((x, y), 0.19, color=ring, fill=False,
                                lw=rlw, zorder=6))
        ax.text(x, y, agent.replace("agent", "A"), fontsize=11,
                fontweight="bold", color="white", ha="center", va="center",
                zorder=7)
        if node_answers and agent in node_answers:
            ax.text(x, y - 0.27, node_answers[agent], fontsize=7.5,
                    color="#cccccc", ha="center", va="top", zorder=7)

    ax.text(0, 1.35, phase_label, fontsize=12, color="white",
            ha="center", va="top", fontweight="bold")
    plt.tight_layout(pad=0)
    return fig


def ans_str(agent, use_after=False):
    src = after if use_after else before
    return "  ".join(f"{q['id']}:{q['value']}" for q in src[agent])


c1, c2, _ = st.columns([1, 1, 1])
with c1:
    speed = st.slider("sec/step", 0.3, 2.5, 1.0, 0.1)
with c2:
    play = st.button("play", use_container_width=True)

st.divider()
col_graph, col_text = st.columns([1, 1])
placeholder = col_graph.empty()
status      = col_text.empty()

with placeholder.container():
    fig = draw(AGENT_NAMES, [], [], {a: ans_str(a) for a in AGENT_NAMES},
               f"Round {round_num} — press play")
    st.pyplot(fig, width='content'); plt.close(fig)

if not play:
    st.stop()

# Phase 1
for i, agent in enumerate(AGENT_NAMES):
    visible = AGENT_NAMES[:i+1]
    with placeholder.container():
        fig = draw(visible, [], [], {a: ans_str(a) for a in visible},
                   f"Phase 1 — Initial answers  (Round {round_num})")
        st.pyplot(fig, width='content'); plt.close(fig)
    lines = []
    for a in visible:
        lines.append(f"**{a}**")
        for q in before[a]:
            lines.append(f"- {q['id']}: {q['value']}")
    status.info("\n\n".join(lines))
    time.sleep(speed)

# Phase 2
shown = []
for idx, edge in enumerate(critique_edges):
    src, tgt, label = edge
    shown.append(edge)
    with placeholder.container():
        fig = draw(AGENT_NAMES, shown, [src],
                   {a: ans_str(a) for a in AGENT_NAMES},
                   f"Phase 2 — Critique  (Round {round_num})",
                   latest_edge_idx=idx)
        st.pyplot(fig, width='content'); plt.close(fig)
    status.warning(f"**{src}** → **{tgt}**\n\n{label}")
    time.sleep(speed)

# Phase 3
for agent in AGENT_NAMES:
    changed = any(v[2] for v in answer_changes[agent].values())
    ans_now = {a: ans_str(a, use_after=(a == agent)) for a in AGENT_NAMES}
    with placeholder.container():
        fig = draw(AGENT_NAMES, critique_edges,
                   [agent] if changed else [],
                   ans_now,
                   f"Phase 3 — Rebuttal  (Round {round_num})")
        st.pyplot(fig, width='content'); plt.close(fig)
    if changed:
        diff = ", ".join(f"{qid}: {b}->{av}"
                         for qid, (b, av, c) in answer_changes[agent].items() if c)
        status.error(f"**{agent}** answer changed: {diff}")
    else:
        status.success(f"**{agent}** answer unchanged")
    time.sleep(speed)

# Phase 4
agreed = supervisor["agreement"]
with placeholder.container():
    fig = draw(AGENT_NAMES, [],
               AGENT_NAMES if agreed else [],
               {a: ans_str(a, use_after=True) for a in AGENT_NAMES},
               f"Phase 4 — {'Consensus' if agreed else 'No consensus'}  (Round {round_num})")
    st.pyplot(fig, width='content'); plt.close(fig)

q_summary = "  |  ".join(
    f"{q}: {v['agent1']}" for q, v in supervisor["answer_map"].items()
)
if agreed:
    status.success(f"Supervisor: agreement — {q_summary}")
else:
    status.error(f"Supervisor: no agreement — {q_summary}")