import base64
import os


def get_b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")


# Paths
base_dir = "/home/sicheng/AgentCAP/docs/figures"
img_a_pareto = get_b64(f"{base_dir}/pairA/pareto_frontier.png")
img_a_strat = get_b64(f"{base_dir}/pairA/strategy_comparison.png")
img_a_cost = get_b64(f"{base_dir}/pairA/cost_per_correct.png")
img_a_esc = get_b64(f"{base_dir}/pairA/escalation_analysis.png")

img_b_pareto = get_b64(f"{base_dir}/pairB/pareto_frontier.png")
img_b_strat = get_b64(f"{base_dir}/pairB/strategy_comparison.png")
img_b_cost = get_b64(f"{base_dir}/pairB/cost_per_correct.png")
img_b_esc = get_b64(f"{base_dir}/pairB/escalation_analysis.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentCAP NeurIPS 2026 Presentation</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --text-color: #f0f0f5;
            --accent-primary: #00d4ff;
            --accent-secondary: #ffd700;
            --table-row-even: rgba(255, 255, 255, 0.03);
            --table-row-odd: rgba(255, 255, 255, 0.08);
            --pareto-bg: rgba(0, 212, 255, 0.15);
        }}
        
        body, html {{
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow: hidden;
        }}

        .slides-container {{
            width: 100%;
            height: 100%;
            position: relative;
        }}

        .slide {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.5s ease-in-out, visibility 0.5s ease-in-out;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
            padding: 4rem 10%;
        }}

        .slide.active {{
            opacity: 1;
            visibility: visible;
        }}

        .slide-content {{
            width: 100%;
            max-width: 1200px;
            display: flex;
            flex-direction: column;
        }}

        h1 {{
            color: var(--accent-primary);
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }}

        h2 {{
            color: var(--accent-primary);
            font-size: 2.5rem;
            margin-bottom: 2rem;
            border-bottom: 2px solid var(--accent-secondary);
            padding-bottom: 0.5rem;
        }}

        h3 {{
            color: var(--accent-secondary);
            font-size: 1.8rem;
        }}

        .subtitle {{
            font-size: 1.5rem;
            color: var(--accent-secondary);
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }}

        ul, ol {{
            font-size: 1.4rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }}

        li {{
            margin-bottom: 0.8rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 1.2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}

        th {{
            background-color: rgba(0, 212, 255, 0.1);
            color: var(--accent-primary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        tr:nth-child(even) {{ background-color: var(--table-row-even); }}
        tr:nth-child(odd) {{ background-color: var(--table-row-odd); }}
        
        tr.pareto {{
            background-color: var(--pareto-bg);
            border-left: 4px solid var(--accent-secondary);
        }}

        .image-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            width: 100%;
            align-items: center;
            justify-items: center;
        }}

        .image-grid img {{
            max-width: 100%;
            max-height: 50vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        }}

        .controls {{
            position: absolute;
            bottom: 2rem;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            z-index: 100;
        }}

        .dots {{
            display: flex;
            gap: 0.8rem;
            margin-bottom: 1rem;
        }}

        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.3);
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .dot:hover {{
            background-color: rgba(255, 255, 255, 0.7);
        }}

        .dot.active {{
            background-color: var(--accent-primary);
            transform: scale(1.2);
            box-shadow: 0 0 10px var(--accent-primary);
        }}

        .slide-indicator {{
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.6);
            font-variant-numeric: tabular-nums;
        }}

        .status-badge {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 1rem;
            font-weight: bold;
            margin-left: 1rem;
            background: rgba(255, 215, 0, 0.2);
            color: var(--accent-secondary);
        }}
        
        .highlight {{ color: var(--accent-secondary); font-weight: bold; }}
        .highlight-blue {{ color: var(--accent-primary); font-weight: bold; }}
    </style>
</head>
<body>

<div class="slides-container" id="slides-container">
    
    <!-- Slide 1 -->
    <div class="slide active">
        <div class="slide-content" style="align-items: center; text-align: center;">
            <h1>When Two Heads Are Cheaper Than One</h1>
            <div class="subtitle">Cost-Optimal Multi-Agent Combinations on Local GPU Clusters</div>
            <h3 style="margin-top: 3rem;">NeurIPS 2026 — AgentCAP Project Update</h3>
        </div>
    </div>

    <!-- Slide 2 -->
    <div class="slide">
        <div class="slide-content">
            <h2>The Problem</h2>
            <ul>
                <li>API pricing doesn't reflect local deployment costs (CapEx vs OpEx).</li>
                <li>Multi-agent strategies have cost-accuracy tradeoffs unexplored on local GPUs.</li>
                <li>No systematic study of >2 model combinations exists.</li>
            </ul>
        </div>
    </div>

    <!-- Slide 3 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Three Core Contributions</h2>
            <ol>
                <li><span class="highlight-blue">CapEx+OpEx Cost Model</span> for local GPU clusters</li>
                <li><span class="highlight-blue">9 multi-agent strategies</span> (6 two-model + 3 novel >2-model)</li>
                <li><span class="highlight-blue">3D Tradeoff Analysis:</span> Accuracy × Cost × Performance</li>
            </ol>
        </div>
    </div>

    <!-- Slide 4 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Cost Model (Contribution 1) <span class="status-badge">⚠️ Proposed, Not Implemented</span></h2>
            <ul>
                <li><strong>Formula:</strong> <span class="highlight">Total_Cost(task) = C_gpu + C_cpu + C_tool + C_idle</span></li>
                <li>CapEx rate table for A100, H100, H200, H20, B200</li>
                <li>3 deployment scenarios:
                    <ul>
                        <li>Idle GPUs</li>
                        <li>Saturated Cluster</li>
                        <li>Cloud Rental</li>
                    </ul>
                </li>
            </ul>
        </div>
    </div>

    <!-- Slide 5 -->
    <div class="slide">
        <div class="slide-content">
            <h2>9 Strategies Overview (Contribution 2)</h2>
            <h3>6 Two-Model Strategies</h3>
            <table style="font-size: 1rem;">
                <tr><th>Strategy</th><th>Mechanism</th><th>Cost Profile</th></tr>
                <tr><td>Cascade</td><td>Small first → escalate if low confidence</td><td>Sequential, low avg</td></tr>
                <tr><td>Adaptive-Cascade</td><td>Dynamic threshold cascade</td><td>Sequential, adaptive</td></tr>
                <tr><td>Vote</td><td>3× small, majority vote</td><td>Parallel, 3× small</td></tr>
                <tr><td>Generate-Verify</td><td>Small generates, large verifies</td><td>Sequential, 1S+1L</td></tr>
                <tr><td>Best-of-N</td><td>N× same model, pick best</td><td>Parallel, N×</td></tr>
                <tr><td>Self-Critique</td><td>Model critiques own output</td><td>Sequential, 2×</td></tr>
            </table>
            <h3>3 Novel >2-Model Strategies</h3>
            <table style="font-size: 1rem;">
                <tr><th>Strategy</th><th>Mechanism</th></tr>
                <tr><td><span class="highlight">Diversity-Gated Cascade ⭐</span></td><td>2 cheap parallel → disagree → escalate</td></tr>
                <tr><td>Cross-Family Vote</td><td>3 different families vote</td></tr>
                <tr><td>3-Tier Cascade</td><td>small → medium → large routing</td></tr>
            </table>
        </div>
    </div>

    <!-- Slide 6 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Experiment Design</h2>
            <ul>
                <li><strong>Hardware:</strong> 4× RTX PRO 6000 Blackwell (~98GB each)</li>
                <li><strong>Serving:</strong> SGLang 0.5.6, all local inference</li>
                <li><strong>Benchmarks:</strong> BigCodeBench (non-agentic) + MCP-Atlas (agentic)</li>
                <li><strong>Models:</strong> Qwen3-4B, Qwen3-30B-A3B (MoE), Qwen3-32B</li>
                <li><strong>Evaluation Pairs:</strong>
                    <ul>
                        <li><span class="highlight">Pair A:</span> Qwen3-4B vs Qwen3-30B-A3B</li>
                        <li><span class="highlight">Pair B:</span> Qwen3-30B-A3B vs Qwen3-32B</li>
                    </ul>
                </li>
            </ul>
        </div>
    </div>

    <!-- Slide 7 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Pair A Results Table (400/400)</h2>
            <table>
                <tr><th>Strategy</th><th>Pass@1</th><th>GPU-s/task</th><th>$/correct</th><th>Pareto?</th></tr>
                <tr class="pareto"><td>Best-of-N-Small (4B×3)</td><td>50%</td><td>104.4</td><td>$0.070</td><td>✅</td></tr>
                <tr class="pareto"><td>Cascade</td><td>44%</td><td>23.9</td><td>$0.018</td><td>✅</td></tr>
                <tr><td>Best-of-N-Large (30B×3)</td><td>44%</td><td>74.7</td><td>$0.057</td><td></td></tr>
                <tr><td>Adaptive-Cascade</td><td>42%</td><td>41.6</td><td>$0.033</td><td></td></tr>
                <tr><td>Vote</td><td>36%</td><td>25.3</td><td>$0.023</td><td></td></tr>
                <tr><td>Generate-Verify</td><td>30%</td><td>25.1</td><td>$0.028</td><td></td></tr>
                <tr><td>Self-Critique-Small</td><td>24%</td><td>52.4</td><td>$0.073</td><td></td></tr>
                <tr><td>Self-Critique-Large</td><td>14%</td><td>23.6</td><td>$0.056</td><td></td></tr>
            </table>
        </div>
    </div>

    <!-- Slide 8 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Pair A: Tradeoffs & Comparisons</h2>
            <div class="image-grid">
                <img src="{img_a_pareto}" alt="Pair A Pareto Frontier">
                <img src="{img_a_strat}" alt="Pair A Strategy Comparison">
            </div>
        </div>
    </div>

    <!-- Slide 9 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Pair A: Cost & Escalation</h2>
            <div class="image-grid">
                <img src="{img_a_cost}" alt="Pair A Cost per Correct">
                <img src="{img_a_esc}" alt="Pair A Escalation Analysis">
            </div>
        </div>
    </div>

    <!-- Slide 10 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Pair A: Escalation Deep Dive</h2>
            <ul>
                <li><strong>Cascade Strategy:</strong> 68% escalation rate</li>
                <li><strong>Non-escalated accuracy:</strong> <span class="highlight">100%</span> (perfect confidence calibration!)</li>
                <li><strong>Escalated accuracy:</strong> 17.6%</li>
            </ul>
            <div style="margin-top: 2rem; padding: 2rem; background: rgba(0, 212, 255, 0.1); border-left: 4px solid var(--accent-primary);">
                <h3>Insight: Small model confidence = correctness</h3>
                <p style="font-size: 1.4rem;">When the small model is confident, it is almost always correct. Escalate only when uncertain.</p>
            </div>
        </div>
    </div>

    <!-- Slide 11 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Pair B Results Table (Partial)</h2>
            <table>
                <tr><th>Strategy</th><th>Done</th><th>Pass@1</th><th>GPU-s/task</th><th>Pareto?</th></tr>
                <tr class="pareto"><td>Cascade</td><td>44/50</td><td>52.3%</td><td>25.6</td><td>✅</td></tr>
                <tr><td>Best-of-N-Small (30B×3)</td><td>50/50</td><td>46.0%</td><td>78.5</td><td></td></tr>
                <tr><td>Adaptive-Cascade</td><td>45/50</td><td>44.4%</td><td>32.7</td><td></td></tr>
                <tr class="pareto"><td>Vote</td><td>43/50</td><td>44.2%</td><td>23.3</td><td>✅</td></tr>
                <tr class="pareto"><td>Generate-Verify</td><td>48/50</td><td>43.8%</td><td>23.0</td><td>✅</td></tr>
                <tr><td>Best-of-N-Large (32B×3)</td><td>5/50</td><td>60.0%*</td><td>420.0</td><td>N/A</td></tr>
            </table>
            <p style="font-size: 1rem; color: #888; font-style: italic;">Note: *only 5 samples, unreliable</p>
        </div>
    </div>

    <!-- Slide 12 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Pair B: Tradeoffs & Comparisons</h2>
            <div class="image-grid">
                <img src="{img_b_pareto}" alt="Pair B Pareto Frontier">
                <img src="{img_b_strat}" alt="Pair B Strategy Comparison">
            </div>
        </div>
    </div>

    <!-- Slide 13 -->
    <div class="slide">
        <div class="slide-content">
            <h2>MCP-Atlas Results (Agentic)</h2>
            <table>
                <tr><th>Strategy</th><th>Pass@1</th><th>Coverage</th></tr>
                <tr><td>Cascade</td><td><span class="highlight-blue">8%</span></td><td>13.6%</td></tr>
                <tr><td>Best-of-N-Small</td><td>2%</td><td>13.2%</td></tr>
            </table>
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(255, 215, 0, 0.1); border-left: 4px solid var(--accent-secondary);">
                <h3>Key Takeaways</h3>
                <ul>
                    <li>Best-of-N dominates BigCodeBench but <span class="highlight">FAILS on agentic tasks</span></li>
                    <li>Cascade is the only viable multi-agent strategy for agentic workloads</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Slide 14 -->
    <div class="slide">
        <div class="slide-content" style="max-height: 80vh; overflow-y: auto;">
            <h2>6 Key Insights</h2>
            <ol>
                <li style="margin-bottom: 1rem;">🏆 <strong>CASCADE IS KING</strong> — Pareto-optimal, $0.018/correct</li>
                <li style="margin-bottom: 1rem;">💰 <strong>CHEAP×3 > EXPENSIVE×1</strong> — BoN-Small 50% beats single large 38%</li>
                <li style="margin-bottom: 1rem;">❌ <strong>SELF-CRITIQUE DESTROYS</strong> — drops accuracy 6-24pp</li>
                <li style="margin-bottom: 1rem;">🔄 <strong>BENCHMARK CHANGES EVERYTHING</strong> — BoN wins coding, Cascade wins agentic</li>
                <li style="margin-bottom: 1rem;">🧬 <strong>SAME-FAMILY CASCADE HELPS</strong> — Pair B 52% > Pair A 44%</li>
                <li style="margin-bottom: 1rem;">✅ <strong>CONFIDENCE = CORRECTNESS</strong> — Non-escalated tasks: 100% correct</li>
            </ol>
        </div>
    </div>

    <!-- Slide 15 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Implementation Status</h2>
            <ul style="list-style-type: none; padding: 0;">
                <li><span style="color: #4cd137;">✅</span> 7 strategy implementations</li>
                <li><span style="color: #4cd137;">✅</span> BigCodeBench evaluator + MCP-Atlas runner</li>
                <li><span style="color: #4cd137;">✅</span> Pair A complete (400/400), Pair B partial (235/300)</li>
                <li style="margin-top: 1.5rem; color: #ff9f43;"><span class="status-badge">⚠️</span> <strong>Cost model code NOT implemented</strong> (paper Contribution 1!)</li>
                <li style="color: #ff9f43;"><span class="status-badge">⚠️</span> >2 model strategies NOT coded</li>
                <li style="color: #ff9f43;"><span class="status-badge">⚠️</span> New model families NOT downloaded</li>
            </ul>
        </div>
    </div>

    <!-- Slide 16 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Next Steps</h2>
            <ul>
                <li><strong>Phase A:</strong> Implement Cost Model <span class="highlight">(no GPU needed)</span></li>
                <li><strong>Phase B:</strong> Code >2 model strategies <span class="highlight">(no GPU needed)</span></li>
                <li><strong>Phase C:</strong> Complete Pair B experiments</li>
                <li><strong>Phase D:</strong> Download new models + cross-family experiments</li>
                <li><strong>Phase E:</strong> >2 model experiments</li>
                <li><strong>Phase F:</strong> System performance experiments</li>
                <li><strong>Phase G:</strong> Final analysis + paper figures</li>
            </ul>
        </div>
    </div>

    <!-- Slide 17 -->
    <div class="slide">
        <div class="slide-content">
            <h2>Research Questions (NeurIPS 2026)</h2>
            <div style="background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 8px; border-left: 4px solid var(--accent-primary);">
                <h3 style="margin-top: 0; color: var(--accent-primary);">RQ1</h3>
                <p style="font-size: 1.4rem;">Which combinations dominate accuracy-cost Pareto frontier?</p>
                
                <h3 style="color: var(--accent-primary);">RQ2</h3>
                <p style="font-size: 1.4rem;">Does cross-family diversity improve the frontier?</p>
                
                <h3 style="color: var(--accent-primary);">RQ3</h3>
                <p style="font-size: 1.4rem;">How do scaling and task type shift optimal strategy?</p>
            </div>
        </div>
    </div>

</div>

<div class="controls">
    <div class="dots" id="dots-container"></div>
    <div class="slide-indicator" id="slide-indicator">1 / 17</div>
</div>

<script>
    const slides = document.querySelectorAll('.slide');
    const dotsContainer = document.getElementById('dots-container');
    const indicator = document.getElementById('slide-indicator');
    let currentSlide = 0;

    // Create dots
    slides.forEach((_, index) => {{
        const dot = document.createElement('span');
        dot.classList.add('dot');
        if (index === 0) dot.classList.add('active');
        dot.addEventListener('click', () => showSlide(index));
        dotsContainer.appendChild(dot);
    }});

    const dots = document.querySelectorAll('.dot');

    function showSlide(n) {{
        slides[currentSlide].classList.remove('active');
        dots[currentSlide].classList.remove('active');
        
        currentSlide = (n + slides.length) % slides.length;
        
        slides[currentSlide].classList.add('active');
        dots[currentSlide].classList.add('active');
        indicator.textContent = (currentSlide + 1) + ' / ' + slides.length;
    }}

    document.addEventListener('keydown', (e) => {{
        if (e.key === 'ArrowRight' || e.key === 'Space') {{
            showSlide(currentSlide + 1);
        }} else if (e.key === 'ArrowLeft') {{
            showSlide(currentSlide - 1);
        }}
    }});

    // Initialize indicator
    indicator.textContent = '1 / ' + slides.length;
</script>

</body>
</html>"""

with open("/home/sicheng/AgentCAP/docs/presentation.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Successfully generated presentation.html")
